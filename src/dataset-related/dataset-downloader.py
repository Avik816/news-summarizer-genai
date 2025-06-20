# For this project download the dataset from HuggingFace.
# The HuggingFace dataset is used because of the following reasons:
#   • The dataset is structured; best for any kind of AI/ML tasks.
#   • The dataset is paired as article and summary.
#   • Where the article is the full news article and summary is the summary of it.

# This script downloads the dataset from the HuggingFace datasets.
# Originally the dataset will be in Hugging Face-optimized binary format using Apache Arrow.
# It will be converted to .csv format for easier uses (ref. dataset-maker.py).


from datasets import load_dataset


dataset = load_dataset('abisee/cnn_dailymail', '3.0.0') # name of the dataset and the version of it

# Saving the datasets to desired location
dataset['train'].to_csv('datasets/train.csv')
dataset['validation'].to_csv('datasets/validation.csv')
dataset['test'].to_csv('datasets/test.csv')

print('Original Datasets saved !')