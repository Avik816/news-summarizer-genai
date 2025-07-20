from torch.nn.utils.rnn import pad_sequence
# from ..utils.tokenizer import get_tokenizer


def collate_fn(batch, tokenizer):
    # tokenizer = get_tokenizer()
    
    input_ids, target_ids = zip(*batch)

    input_ids_padded = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    target_ids_padded = pad_sequence(
        target_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id
    )
    
    attention_mask = (input_ids_padded != tokenizer.pad_token_id).long()

    return input_ids_padded, attention_mask, target_ids_padded