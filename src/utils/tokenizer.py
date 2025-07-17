from transformers import T5Tokenizer
from ..CONFIG import MODEL_NAME


def get_tokenizer():
    # Downloading and setting up the tokenizer
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    return tokenizer