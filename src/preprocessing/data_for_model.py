from ..CONFIG import TRAIN_SIZE, VAL_SIZE, TEST_SIZE, CLEANED_FULL_DATASET_DIR, TRAIN_DIR, VAL_DIR, TEST_DIR
import polars, os


def make_dataset_for_model():
    dataset = polars.read_csv(CLEANED_FULL_DATASET_DIR)

    train = dataset[:int(dataset.shape[0] * TRAIN_SIZE), :]
    val = dataset[int(train.shape[0]):int(train.shape[0] + int(dataset.shape[0] * VAL_SIZE)), :]
    test = dataset[int(val.shape[0]):int(val.shape[0] + int(dataset.shape[0] * TEST_SIZE)), :]
    
    os.mkdir('data')

    train.write_csv(TRAIN_DIR)
    val.write_csv(VAL_DIR)
    test.write_csv(TEST_DIR)

    print('Dataset splitted and saved in {data}.')