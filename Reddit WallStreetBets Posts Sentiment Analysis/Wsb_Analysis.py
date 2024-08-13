import pandas as pd
import string
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Load the CSV file
file_path = r'C:\Users\Andrew\Desktop\NEU\5100 foundation for AI\Final\Reddit WallStreetBets Posts\reddit_wsb.csv'
reddit_data = pd.read_csv(file_path)

# Fill missing 'body' content with the 'title'
reddit_data['body'] = reddit_data['body'].fillna(reddit_data['title'])

# Function to preprocess the text
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    return text

# Apply the preprocessing function to the 'body' column
reddit_data['processed_body'] = reddit_data['body'].apply(preprocess_text)

# Load the pre-trained model and tokenizer
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Function to get sentiment score using BERT
def get_transformer_sentiment_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)
    sentiment_score = probabilities[:, 1].item()  # Index 1 is for positive sentiment
    return sentiment_score

# Apply the sentiment analysis function to the entire processed text
reddit_data['transformer_sentiment_score'] = reddit_data['processed_body'].apply(get_transformer_sentiment_score)

# Normalize the score and number of comments
reddit_data['normalized_score'] = (reddit_data['score'] - reddit_data['score'].min()) / (reddit_data['score'].max() - reddit_data['score'].min())
reddit_data['normalized_comms_num'] = (reddit_data['comms_num'] - reddit_data['comms_num'].min()) / (reddit_data['comms_num'].max() - reddit_data['comms_num'].min())

# Combine the sentiment score, normalized score, and normalized comments into a combined score
reddit_data['combined_score'] = (
    reddit_data['transformer_sentiment_score'] + 
    reddit_data['normalized_score'] + 
    reddit_data['normalized_comms_num']
) / 3  # Average of the three components

# Save the output to a new CSV file with only relevant columns
output_file_path = r'C:\Users\Andrew\Desktop\NEU\5100 foundation for AI\Final\Reddit WallStreetBets Posts\reddit_wsb_combined_scores.csv'
reddit_data[['body', 'score', 'comms_num', 'transformer_sentiment_score', 'combined_score']].to_csv(output_file_path, index=False)

# Display the first few rows of the newly created file with combined scores
print(reddit_data[['body', 'score', 'comms_num', 'transformer_sentiment_score', 'combined_score']].head())
