# Paths
RAW_TRAIN_DIR = 'raw datasets/train.csv'
RAW_VAL_DIR = 'raw datasets/val.csv'
RAW_TEST_DIR = 'raw datasets/test.csv'
RAW_FULL_DATASET_DIR = 'raw datasets/Full news dataset.csv'
CLEANED_FULL_DATASET_DIR = 'raw datasets/Cleaned news dataset.csv'
MODEL_PATH = 'saved_model/T5_small_v1'
TOKENIZER_PATH = 'saved_tokenizer'
CHECKPOINT_PATH = 'saved_checkpoints/T5_small_v1'
CSV_LOGGER_PATH = 'T5_small'
LOSS_CURVE_PATH = 'plots'

TRAIN_DIR = 'data/train.csv'
VAL_DIR = 'data/val.csv'
TEST_DIR = 'data/test.csv'

# Model Parameters
TRAIN_SIZE = 0.80
VAL_SIZE = 0.10
TEST_SIZE = 0.10

# Model Hyper-parameters
MODEL_NAME = 't5-small'
MAX_INPUT_LENGTH = 512
MAX_TARGET_LENGTH = 128
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 5e-5
PATIENCE = 3
PATIENCE_ON_RLR = 2
FACTOR = 0.5
FREQUENCY = 1