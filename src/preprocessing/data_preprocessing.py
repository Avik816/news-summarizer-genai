'''
Objectives:
a. Check for missing values
b. Remove unwanted columns
c. Remove any unwanted spaces from the article column
d. Replace curly apostrophes and special quotes with standard UTF-8 apostrophes (e.g., ’ → ')
e. Remove apostrophes if necessary for normalization
f. Remove author name and publishing/update date as they are noisy data
g. Remove legal disclaimers, copyright notices, and redistribution warnings
h. Remove attribution or contribution credits (e.g., “CNN’s John Doe contributed to this report.”)
i. Remove email addresses and social media handles (e.g., @CNN)
j. Remove hyperlinks and web URLs from the text
k. Remove source tags and image credits like (Reuters), (AP), (Getty)
l. Remove redundant phrases like “Scroll down for video”, “Read more at”, “Related Articles”
m. Remove text enclosed in square or curly brackets (e.g., [Editor’s note])
n. Remove HTML tags if present
o. Normalize multiple dots, pipes, and extra spaces or newlines
'''


import polars
from ..CONFIG import RAW_FULL_DATASET_DIR, CLEANED_FULL_DATASET_DIR
import re


def clean_article(text):
    # Normalize apostrophes
    text = text.replace("’", "'").replace("‘", "'")

    # Remove "By Author" lines
    text = re.sub(r'by\s+[\w\s\.,]+(?:\n|\|)', '', text, flags=re.IGNORECASE)

    # Remove PUBLISHED/UPDATED timestamps
    text = re.sub(r'(PUBLISHED|UPDATED|Last updated).*?(\n|\.|$)', '', text, flags=re.IGNORECASE)

    # Remove legal disclaimers, copyrights, and redistribution warnings
    text = re.sub(r'(e-?mail to a friend.*?|all rights reserved.*?|this material may not be published.*?|copyright \d{4}.*?|©\s*\d{4}.*?)($|\n|\.)', '', text, flags=re.IGNORECASE)

    # Remove email addresses and Twitter handles
    text = re.sub(r'\S+@\S+', '', text)      # Emails
    text = re.sub(r'@\w+', '', text)         # Twitter handles

    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)

    # Remove source tags and image captions
    text = re.sub(r'\(Reuters\)|\(AP\)|\(Getty.*?\)', '', text)
    text = re.sub(r'Scroll down for .*?(\.|\n)', '', text, flags=re.IGNORECASE)

    # Remove redundant phrases
    text = re.sub(r'(Read more at|Full story|Related Articles).*?(\n|$)', '', text, flags=re.IGNORECASE)

    # Remove attribution/contribution credits
    text = re.sub(r'(contributed to this report.*?|with (additional )?reporting by.*?|reporting by .*?editing by .*?)\.?', '', text, flags=re.IGNORECASE)

    # Remove text in square or curly brackets
    text = re.sub(r'\[.*?\]|\{.*?\}', '', text)

    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)

    # Normalize whitespace and punctuation
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'\s*\|\s*', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)
    text = re.sub(r'\n+', ' ', text)

    return text.strip()

def preprocess_dataset():
    dataset = polars.read_csv(RAW_FULL_DATASET_DIR)

    # Missing values
    print(dataset.null_count())

    # Removing unwanted columns
    dataset = dataset.drop('id')

    # Performing objectives 'c' to 'o'
    dataset = dataset.with_columns(
    polars.col('article').
    map_elements(
        function= lambda t: clean_article(t),
        return_dtype=polars.Utf8 
    ).alias('article'))

    dataset.write_csv(CLEANED_FULL_DATASET_DIR)
    print('Dataset cleaned and saved in {raw datasets}')