import warnings
import src


warnings.filterwarnings('ignore')

if __name__ == '__main__':
    """print('DOWNLOADING DATASET ...')
    src.download_dataset()

    print('MAKING FULL DATASET ...')
    src.make_full_dataset()

    print('PREPROCESSING DATASET ...')
    src.preprocess_dataset()"""

    print('MAKING DATASET FOR THE MODEL ...')
    src.make_dataset_for_model()

    print('MODEL TRAINING ...')
    src.train_T5Small_model()