import polars
from ..CONFIG import RAW_TRAIN_DIR, RAW_VAL_DIR, RAW_TEST_DIR, RAW_FULL_DATASET_DIR
import os


def make_full_dataset():
    train = polars.read_csv(RAW_TRAIN_DIR)
    val = polars.read_csv(RAW_VAL_DIR)
    test = polars.read_csv(RAW_TEST_DIR)

    dataset = train.vstack(val).vstack(test)

    print(dataset.shape)

    dataset.write_csv(RAW_FULL_DATASET_DIR)

    print('Full raw dataset saved in {raw dataset} folder.')