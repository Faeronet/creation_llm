"""
Inference pipeline: question -> retrieval -> answerability -> generation or refusal -> post-check.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch

from modules.answerability_model import AnswerabilityModel
from modules.constants import DEFAULT_REFUSAL_MESSAGE, PARTIAL_REFUSAL_SUFFIX
from modules.generation import generate
from modules.logger import get_logger
from modules.model_config import LMConfig
from modules.modeling_decoder_lm import DecoderLM
from modules.postcheck import answer_supported_by_context
from modules.prompt_builder import build_prompt
from modules.retriever import Retriever
from modules.tokenizer_utils import decode, load_tokenizer

logger = get_logger(__name__)


class InferencePipeline:
    """
    Full pipeline: load tokenizer, LM, answerability model, retriever;
    run question -> retrieve -> answerable? -> generate -> post-check -> response or refusal.
    """

    def __init__(
        self,
        tokenizer_dir: Optional[Path] = None,
        lm_dir: Optional[Path] = None,
        answerability_dir: Optional[Path] = None,
        retrieval_dir: Optional[Path] = None,
        top_k: int = 5,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        use_fp16: bool = True,
        answerability_threshold: float = 0.5,
    ):
        from modules.paths import MODEL_ANSWERABILITY, MODEL_LM, MODEL_RETRIEVAL, MODEL_TOKENIZER

        tokenizer_dir = tokenizer_dir or MODEL_TOKENIZER
        lm_dir = lm_dir or MODEL_LM
        answerability_dir = answerability_dir or MODEL_ANSWERABILITY
        retrieval_dir = retrieval_dir or MODEL_RETRIEVAL

        self.sp = load_tokenizer(tokenizer_dir)
        self.retriever = Retriever(retrieval_dir)
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.use_fp16 = use_fp16
        self.answerability_threshold = answerability_threshold

        config_path = lm_dir / "config.json"
        if config_path.exists():
            self.lm_config = LMConfig.load(config_path)
        else:
            self.lm_config = LMConfig()
        self.lm = DecoderLM(self.lm_config)
        state = torch.load(lm_dir / "pytorch_model.pt", map_location="cpu", weights_only=True)
        if "model_state_dict" in state:
            state = state["model_state_dict"]
        self.lm.load_state_dict(state, strict=False)
        self.lm.eval()

        self.answerability_model = AnswerabilityModel(self.lm_config, freeze_lm=True)
        self.answerability_model.load_lm_weights(lm_dir / "pytorch_model.pt")
        head_path = answerability_dir / "head.pt"
        if head_path.exists():
            self.answerability_model.head.load_state_dict(
                torch.load(head_path, map_location="cpu", weights_only=True)
            )
        self.answerability_model.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lm = self.lm.to(self._device)
        self.answerability_model = self.answerability_model.to(self._device)

    def _is_answerable(self, question: str, context_text: str) -> bool:
        """Run answerability classifier on question + context."""
        text = (context_text + " " + question).strip()
        ids = self.sp.encode(text, add_bos=True, add_eos=True, out_type=int)[:512]
        input_ids = torch.tensor([ids], dtype=torch.long, device=self._device)
        with torch.no_grad():
            logits, _ = self.answerability_model(input_ids)
        probs = torch.softmax(logits, dim=-1)
        return probs[0, 1].item() >= self.answerability_threshold

    def run(self, question: str) -> Tuple[str, bool]:
        """
        Run full pipeline.

        Returns:
            (response_text, was_refusal). If was_refusal is True, response_text is the refusal message.
        """
        question = question.strip()
        if not question:
            return DEFAULT_REFUSAL_MESSAGE, True

        chunks = self.retriever.retrieve(question, top_k=self.top_k)
        if not chunks:
            return DEFAULT_REFUSAL_MESSAGE, True
        context_text = "\n\n".join(text for _, text, _ in chunks)
        if not self._is_answerable(question, context_text):
            return DEFAULT_REFUSAL_MESSAGE, True

        prompt_str = build_prompt(question, chunks)
        input_ids = self.sp.encode(prompt_str, add_bos=True, add_eos=False, out_type=int)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self._device)
        if self.use_fp16:
            self.lm = self.lm.half()
        output_ids = generate(
            self.lm,
            input_tensor,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.sp.eos_id(),
            pad_token_id=self.sp.pad_id(),
            temperature=self.temperature,
            do_sample=True,
            use_fp16=self.use_fp16,
        )
        generated = output_ids[0].tolist()
        new_tokens = generated[len(input_ids):]
        answer = decode(self.sp, new_tokens, skip_special_tokens=True).strip()
        if not answer:
            return DEFAULT_REFUSAL_MESSAGE, True
        if not answer_supported_by_context(answer, chunks):
            return DEFAULT_REFUSAL_MESSAGE, True
        return answer, False
