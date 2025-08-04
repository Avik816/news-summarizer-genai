# from ..utils.tokenizer import get_tokenizer
from ..CONFIG import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
import pandas as pd
from torch.utils.data import IterableDataset
import random


"""class StreamingNewsSummaryDataset(IterableDataset):
    def __init__(self, file_path, tokenizer):
        self.file_path = file_path
        self.tokenizer = tokenizer

    def parse(self):
        # tokenizer = get_tokenizer()

        for chunk in pd.read_csv(self.file_path, chunksize=1):
            # Taking the inputs from the dataset
            article = chunk.iloc[0]['article']
            summary = chunk.iloc[0]['highlights']

            # Formatting the input to the actual text input format for T5 model
            input_text = 'summarize: ' + article

            # Encoding the data for model training
            input_ids = self.tokenizer.encode(
                input_text, truncation=True,
                max_length=MAX_INPUT_LENGTH,
                return_tensors='pt'
            ).squeeze(0)
            
            target_ids = self.tokenizer.encode(
                summary,
                truncation=True,
                max_length=MAX_TARGET_LENGTH,
                return_tensors='pt'
            ).squeeze(0)
            
            yield input_ids, target_ids

    def __iter__(self):
        return self.parse()"""

class StreamingNewsSummaryDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, buffer_size=1000):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size

    def parse(self):
        buffer = []

        # Read a larger chunk of data at once
        df = pd.read_csv(self.file_path)

        for row in df.itertuples(index=False):
            article = row.article
            summary = row.highlights

            input_text = 'summarize: ' + article
            input_ids = self.tokenizer.encode(
                input_text,
                truncation=True,
                max_length=MAX_INPUT_LENGTH,
                return_tensors='pt'
            ).squeeze(0)

            target_ids = self.tokenizer.encode(
                summary,
                truncation=True,
                max_length=MAX_TARGET_LENGTH,
                return_tensors='pt'
            ).squeeze(0)

            buffer.append((input_ids, target_ids))

            if len(buffer) >= self.buffer_size:
                random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()

        # Yield remaining items
        random.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    def __iter__(self):
        return self.parse()
