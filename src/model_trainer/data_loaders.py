from ..utils.tokenizer import get_tokenizer
from ..CONFIG import MODEL_NAME, MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
import pandas as pd
from torch.utils.data import IterableDataset
from torch.nn.utils.rnn import pad_sequence


class StreamingNewsSummaryDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def parse(self):
        tokenizer = get_tokenizer()
        
        for chunk in pd.read_csv(self.file_path, chunksize=1):
            # Taking the inputs from the dataset
            article = chunk.iloc[0]['article']
            summary = chunk.iloc[0]['highlights']

            # Formatting the input to the actual text input format for T5 model
            input_text = 'summarize: ' + article

            # Encoding the data for model training
            input_ids = tokenizer.encode(
                input_text, truncation=True,
                max_length=MAX_INPUT_LENGTH,
                return_tensors='pt'
            ).squeeze(0)
            
            target_ids = tokenizer.encode(
                summary,
                truncation=True,
                max_length=MAX_TARGET_LENGTH,
                return_tensors='pt'
            ).squeeze(0)
            
            yield input_ids, target_ids

    def __iter__(self):
        return self.parse()


def collate_fn(batch):
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