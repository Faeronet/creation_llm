"""
Project-wide constants: special tokens, data file names, JSONL field keys.
"""

# Special tokens for tokenizer and model
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
USER_TOKEN = "<|user|>"
ASSISTANT_TOKEN = "<|assistant|>"
SYSTEM_TOKEN = "<|system|>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
CHAT_SPECIAL_TOKENS = [SYSTEM_TOKEN, USER_TOKEN, ASSISTANT_TOKEN]

# Data file names (relative to data/)
FILE_BOOK_CLEAN = "01_book_clean.txt"
FILE_TOKENIZER_CORPUS = "02_tokenizer_corpus.txt"
FILE_ANGELS_STRUCTURED = "03_angels_structured.jsonl"
FILE_RETRIEVAL_CHUNKS = "04_retrieval_chunks.jsonl"
FILE_SFT_CHAT_TRAIN = "05_sft_chat_train.jsonl"
FILE_SFT_CHAT_VAL = "06_sft_chat_val.jsonl"
FILE_ANSWERABILITY_TRAIN = "07_answerability_train.jsonl"
FILE_ANSWERABILITY_VAL = "08_answerability_val.jsonl"
FILE_SYSTEM_PROMPT = "09_system_prompt.txt"
FILE_RECOMMENDED_CONFIG = "10_recommended_training_config.yaml"
FILE_DATASET_README = "README.md"
FILE_STATS = "stats.json"

# JSONL field keys (support multiple possible names)
RETRIEVAL_CHUNK_ID_KEYS = ("chunk_id", "id")
RETRIEVAL_CHUNK_TEXT_KEYS = ("text", "content")
SFT_MESSAGES_KEY = "messages"
SFT_PROMPT_KEY = "prompt"
SFT_COMPLETION_KEY = "completion"
MESSAGE_ROLE_KEY = "role"
MESSAGE_CONTENT_KEY = "content"
ANSWERABILITY_QUESTION_KEYS = ("question", "query")
ANSWERABILITY_LABEL_KEYS = ("label", "answerable")
ANSWERABILITY_LABEL_POSITIVE = "answerable"
ANSWERABILITY_LABEL_NEGATIVE = "not_answerable"

# Default refusal message
DEFAULT_REFUSAL_MESSAGE = "В книге нет данных для ответа на этот вопрос."
PARTIAL_REFUSAL_SUFFIX = " По остальным аспектам вопроса в книге данных нет."
