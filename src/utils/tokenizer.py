import os
from ..CONFIG import TOKENIZER_PATH, MODEL_NAME
from transformers import T5Tokenizer


def get_tokenizer():
    # Downloading and setting up the tokenizer
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = T5Tokenizer.from_pretrained(TOKENIZER_PATH)
    else:
        tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
        tokenizer.save_pretrained(TOKENIZER_PATH)

    return tokenizer