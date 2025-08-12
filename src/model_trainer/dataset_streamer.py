# from ..utils.tokenizer import get_tokenizer
from ..CONFIG import MAX_INPUT_LENGTH, MAX_TARGET_LENGTH
import pandas as pd
import torch
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
        return self.parse()

class StreamingNewsSummaryDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, shuffle, buffer_size=1000):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.buffer_size = buffer_size
        self.shuffle = shuffle

    def parse(self):
        buffer = []
        df = pd.read_csv(self.file_path)

        for row in df.itertuples(index=False):
            input_text = 'summarize: ' + row.article
            input_ids = self.tokenizer.encode(
                input_text,
                truncation=True,
                max_length=MAX_INPUT_LENGTH,
                return_tensors='pt'
            ).squeeze(0)
            target_ids = self.tokenizer.encode(row.highlights, truncation=True, max_length=MAX_TARGET_LENGTH, return_tensors='pt').squeeze(0)

            buffer.append((input_ids, target_ids))

            if len(buffer) >= self.buffer_size:
                if self.shuffle:
                    random.shuffle(buffer)
                while buffer:
                    yield buffer.pop()

        # Final remaining items
        if self.shuffle:
            random.shuffle(buffer)
        while buffer:
            yield buffer.pop()

    def __iter__(self):
        return self.parse()"""




"""class StreamingShuffleNewsSummaryDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, shuffle=True, buffer_size=1000):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_input_len = MAX_INPUT_LENGTH
        self.max_target_len = MAX_TARGET_LENGTH
        self.shuffle = shuffle  # True for train, False for val
        self.buffer_size = buffer_size
        self.epoch = 0  # Will be updated by set_epoch()

    def set_epoch(self, epoch: int):
        # Lightning will call this every epoch when using multiple workers
        self.epoch = epoch

    def parse(self):
        # Using different RNG per epoch for train; fixed seed for val
        rng = random.Random(self.epoch if self.shuffle else 0)
        buffer = []

        # Stream CSV row-by-row
        df_iter = pd.read_csv(self.file_path, chunksize=1)
        for chunk in df_iter:
            row = chunk.iloc[0]

            input_text = "summarize: " + row.article
            input_ids = self.tokenizer.encode(
                input_text,
                truncation=True,
                max_length=self.max_input_len,
                return_tensors="pt"
            ).squeeze(0)

            target_ids = self.tokenizer.encode(
                row.highlights,
                truncation=True,
                max_length=self.max_target_len,
                return_tensors="pt"
            ).squeeze(0)

            buffer.append((input_ids, target_ids))

            if len(buffer) >= self.buffer_size:
                if self.shuffle:
                    rng.shuffle(buffer)  # shuffle only if shuffle=True
                while buffer:
                    yield buffer.pop()

        # Handle remaining items
        if buffer:
            if self.shuffle:
                rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()

    def __iter__(self):
        return self.parse()


class InMemoryPreTokenizedBufferShuffleDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, shuffle=True, buffer_size=1000):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_input_len = MAX_INPUT_LENGTH
        self.max_target_len = MAX_TARGET_LENGTH
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.epoch = 0

        # 1. Load full dataset into RAM
        df = pd.read_csv(self.file_path)

        # 2. Pre-tokenize all rows (done only once!)
        self.data = []
        for row in df.itertuples(index=False):
            input_ids = self.tokenizer.encode(
                "summarize: " + row.article,
                truncation=True,
                max_length=self.max_input_len,
                return_tensors="pt"
            ).squeeze(0)

            target_ids = self.tokenizer.encode(
                row.highlights,
                truncation=True,
                max_length=self.max_target_len,
                return_tensors="pt"
            ).squeeze(0)

            self.data.append((input_ids, target_ids))

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def parse(self):
        rng = random.Random(self.epoch if self.shuffle else 0)
        buffer = []

        for item in self.data:
            buffer.append(item)

            if len(buffer) >= self.buffer_size:
                if self.shuffle:
                    rng.shuffle(buffer)
                while buffer:
                    yield buffer.pop()

        # Handle leftovers
        if buffer:
            if self.shuffle:
                rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()

    def __iter__(self):
        return self.parse()"""


class InMemoryPreTokenizedBufferShuffleDataset(IterableDataset):
    def __init__(self, file_path, tokenizer, shuffle=True, buffer_size=1000):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.max_input_len = MAX_INPUT_LENGTH
        self.max_target_len = MAX_TARGET_LENGTH
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.epoch = 0

        # 1. Load full dataset into RAM
        df = pd.read_csv(self.file_path)

        # 2. Prepare lists for batch tokenization
        articles = ["summarize: " + a for a in df["article"].tolist()]
        summaries = df["highlights"].tolist()

        # 3. Batch tokenize inputs
        input_batch = tokenizer(
            articles,
            truncation=True,
            max_length=self.max_input_len,
            padding=False,
            return_tensors=None  # keep as list of lists for flexibility
        )

        # 4. Batch tokenize targets
        target_batch = tokenizer(
            summaries,
            truncation=True,
            max_length=self.max_target_len,
            padding=False,
            return_tensors=None
        )

        # 5. Store pre-tokenized data as tensors
        self.data = [
            (torch.tensor(input_ids, dtype=torch.long),
             torch.tensor(target_ids, dtype=torch.long))
            for input_ids, target_ids in zip(input_batch["input_ids"], target_batch["input_ids"])
        ]

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def parse(self):
        rng = random.Random(self.epoch if self.shuffle else 0)
        buffer = []

        for item in self.data:
            buffer.append(item)
            if len(buffer) >= self.buffer_size:
                if self.shuffle:
                    rng.shuffle(buffer)
                while buffer:
                    yield buffer.pop()

        # Leftover items
        if buffer:
            if self.shuffle:
                rng.shuffle(buffer)
            while buffer:
                yield buffer.pop()

    def __iter__(self):
        return self.parse()

