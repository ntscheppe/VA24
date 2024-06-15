from transformers import pipeline
import pandas as pd
from tqdm import tqdm

def analyze_sentiments(input_csv, output_csv, model_name="cardiffnlp/twitter-roberta-base-sentiment-latest", max_token_length=512):
    sentiment_pipeline = pipeline(model=model_name)

    # Read the input CSV file
    Reddit_data = pd.read_csv(input_csv)

    # Initialize new columns
    Reddit_data['BERT_class'] = "Neutral"
    Reddit_data['BERT_Compound'] = 0.0
    Reddit_data['BERT_label'] = 0.0

    for i in tqdm(range(len(Reddit_data['cleaned_comment_body'])), desc="Processing Sentiments"):
        comment = Reddit_data['cleaned_comment_body'].iloc[i][:max_token_length]
        sentiment_dict = sentiment_pipeline(comment)
        Reddit_data.loc[i, 'BERT_class'] = sentiment_dict[0]['label'].upper()
        Reddit_data.loc[i, 'BERT_Compound'] = sentiment_dict[0]['score']
        
        # Apply conditions to update 'BERT_label'
        if Reddit_data.loc[i, 'BERT_class'] == 'POSITIVE':
            Reddit_data.loc[i, 'BERT_label'] = 3.0 - Reddit_data.loc[i, 'BERT_Compound']
        elif Reddit_data.loc[i, 'BERT_class'] == 'NEUTRAL':
            Reddit_data.loc[i, 'BERT_label'] = 2.0 - float(Reddit_data.loc[i, 'BERT_Compound'])
        elif Reddit_data.loc[i, 'BERT_class'] == 'NEGATIVE':
            Reddit_data.loc[i, 'BERT_label'] = 1.0 - float(Reddit_data.loc[i, 'BERT_Compound'])

    # Remove rows with 'BERT_Compound' equal to 0.0
    Reddit_data = Reddit_data[Reddit_data['BERT_Compound'] != 0.0]
    
    # Save the result to the output CSV file
    Reddit_data.to_csv(output_csv, index=False)

    # Read the output CSV file
    data = pd.read_csv(output_csv)

    # Calculate percentage for each sentiment class
    sentiment_percentage = data["BERT_class"].value_counts(normalize=True) * 100
    print("Sentiment Percentage:")
    print(sentiment_percentage)

    # Calculate mean for each sentiment class
    mean_sentiments = data.groupby('BERT_class')['BERT_label'].mean()
    print("Mean Sentiments:")
    print(mean_sentiments)


# Example usage:
analyze_sentiments(
    input_csv='subreddits_datafiles/processed_datafiles/cleaned_data1.csv', 
    output_csv='subreddits_datafiles/processed_datafiles_sentiment/sentiment_all_subreddits_data.csv'
)
