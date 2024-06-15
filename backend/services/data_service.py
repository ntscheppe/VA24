import pandas as pd
from pathlib import Path


def load_data(file_path: str) -> pd.DataFrame:
    """
    This function loads the data from a .csv file and returns it as a pandas DataFrame.
    """
    return pd.read_csv(file_path)


def get_posts_per_day(state, start_date, end_date):
    """
    This function returns the number of posts per day for a given state and date range.
    """
    file_path = Path.cwd() / 'subreddits_datafiles\processed_datafiles_sentiment\sentiment_all_subreddits_data.csv'
    df = load_data(file_path)
    df = df[(df['state'] == state) & (df['date'] >= start_date) & (df['date'] <= end_date)]
    return df

