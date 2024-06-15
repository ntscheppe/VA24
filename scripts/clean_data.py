import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from deep_translator import GoogleTranslator
from pathlib import Path
from random_word import RandomWords

class DataCleaner:
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.cleaned_data = None

    def clean_data(self):
        """
        This method cleans the data by removing special characters, digits, and stopwords.
        """
        # Anonymize usernames
        self.anonymize_usernames()

        # Remove special characters and digits
        # Clean 'body' column
        self.data['cleaned_body'] = self.data['body'].apply(lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))
        self.data['cleaned_body'] = self.data['cleaned_body'].apply(lambda x: re.sub(r'\d+', '', x))

        # Clean 'comment_body' column
        self.data['cleaned_comment_body'] = self.data['comment_body'].apply(
            lambda x: re.sub(r'[^a-zA-Z\s]', '', str(x)))
        self.data['cleaned_comment_body'] = self.data['cleaned_comment_body'].apply(lambda x: re.sub(r'\d+', '', x))

        # Remove stopwords in german
        nltk.download('stopwords')
        nltk.download('punkt')
        stop_words = set(stopwords.words('german'))
        self.data['cleaned_body'] = self.data['cleaned_body'].apply(
            lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

        self.data['cleaned_comment_body'] = self.data['cleaned_comment_body'].apply(
            lambda x: ' '.join([word for word in word_tokenize(x) if word.lower() not in stop_words]))

        # Translate the text to english
        self.data['cleaned_body'] = self.data['cleaned_body'].apply(
            lambda x: GoogleTranslator(source='auto', target='en').translate(x) if not x.isascii() else x)

        self.data['cleaned_comment_body'] = self.data['cleaned_comment_body'].apply(
            lambda x: GoogleTranslator(source='auto', target='en').translate(x) if not x.isascii() else x)
        self.cleaned_data = self.data

    def anonymize_usernames(self):
        """
        This method replaces Reddit usernames with random unique words.
        """
        unique_usernames = pd.concat([self.data['author_name'], self.data['comment_author_name']]).unique()
        random_words = self.generate_random_words(len(unique_usernames))

        username_mapping = dict(zip(unique_usernames, random_words))

        self.data['author_name'] = self.data['author_name'].map(username_mapping)
        self.data['comment_author_name'] = self.data['comment_author_name'].map(username_mapping)

    def generate_random_words(self, n):
        """
        Generate a list of n unique random words.
        """
        r = RandomWords()
        random_words = set()

        while len(random_words) < n:
            random_word = r.get_random_word()
            if random_word:  # Ensure the word is not None
                random_words.add(random_word.capitalize())

        return list(random_words)

    def save_cleaned_data(self, file_path: str):
        """
        This method saves the cleaned data to a .csv file.
        """
        self.cleaned_data.to_csv(file_path, index=False)


if __name__ == "__main__":
    dataset_filename = "subreddits_datafiles/europe_politik_posts.csv"
    # target_filename = "subreddits_datafiles/processed_datafiles/cleaned_data1.csv"
    target_folder = Path.cwd() / "subreddits_datafiles/processed_datafiles" / "cleaned_data1.csv"
    dataset_path = Path.cwd() / dataset_filename
    data = pd.read_csv(dataset_path)
    cleaner = DataCleaner(data)
    cleaner.clean_data()
    cleaner.save_cleaned_data(target_folder)