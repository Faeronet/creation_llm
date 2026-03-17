"""
Autoregressive generation: greedy or sampling with temperature, EOS and max_new_tokens.
Supports FP16 for inference.
"""

from typing import List, Optional

import torch

from modules.model_config import LMConfig
from modules.modeling_decoder_lm import DecoderLM


def generate(
    model: DecoderLM,
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_new_tokens: int = 128,
    eos_token_id: Optional[int] = None,
    pad_token_id: Optional[int] = None,
    temperature: float = 1.0,
    do_sample: bool = True,
    use_fp16: bool = False,
) -> torch.Tensor:
    """
    Autoregressive generation.

    Args:
        model: DecoderLM model.
        input_ids: (batch, seq_len) prompt token ids.
        attention_mask: (batch, seq_len) optional.
        max_new_tokens: Maximum number of tokens to generate.
        eos_token_id: Stop when this token is generated. If None, use config.
        pad_token_id: Padding token. If None, use config.
        temperature: Sampling temperature (ignored if do_sample=False).
        do_sample: If True, sample; else greedy.
        use_fp16: Use half precision for generation.

    Returns:
        Generated token ids (batch, seq_len + num_generated), including input prompt.
    """
    config = model.config
    eos_token_id = eos_token_id if eos_token_id is not None else config.eos_token_id
    pad_token_id = pad_token_id if pad_token_id is not None else config.pad_token_id

    if use_fp16:
        model = model.half()

    model.eval()
    batch_size = input_ids.shape[0]
    device = input_ids.device

    with torch.no_grad():
        generated = input_ids.clone()
        past_key_values = None
        for _ in range(max_new_tokens):
            if generated.shape[1] > config.max_position_embeddings:
                # Keep only last max_position_embeddings tokens as context
                context = generated[:, -config.max_position_embeddings:]
            else:
                context = generated

            if attention_mask is not None and attention_mask.shape[1] == generated.shape[1]:
                mask = attention_mask
            else:
                mask = torch.ones_like(context, dtype=torch.long, device=device)

            logits, _ = model(context, attention_mask=mask)
            next_token_logits = logits[:, -1, :]

            if temperature > 0 and do_sample:
                next_token_logits = next_token_logits / temperature
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = torch.cat([generated, next_token], dim=1)

            if (next_token == eos_token_id).all():
                break

    return generated


def generate_single(
    model: DecoderLM,
    input_ids: torch.Tensor,
    max_new_tokens: int = 128,
    eos_token_id: Optional[int] = None,
    temperature: float = 0.8,
    do_sample: bool = True,
    use_fp16: bool = False,
) -> List[int]:
    """
    Generate for a single sequence (batch size 1). Returns list of token ids (prompt + generated).
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    out = generate(
        model,
        input_ids,
        attention_mask=None,
        max_new_tokens=max_new_tokens,
        eos_token_id=eos_token_id,
        temperature=temperature,
        do_sample=do_sample,
        use_fp16=use_fp16,
    )
    return out[0].tolist()
