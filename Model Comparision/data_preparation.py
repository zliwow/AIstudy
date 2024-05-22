'''
CS 6210 Zihao Li
Assignment 1 Part 2, step 1
'''
import nltk
from nltk.corpus import movie_reviews
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import random
import pickle

# Download necessary NLTK data
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load and shuffle the movie reviews dataset
# This dataset contains text files categorized as 'pos' (positive) and 'neg' (negative)
documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)  # Shuffle the documents to ensure a balanced distribution

# Initialize stopwords and lemmatizer
stop_words = set(stopwords.words('english'))  # Load the set of stop words
lemmatizer = WordNetLemmatizer()  # Initialize the lemmatizer

# Preprocess the dataset
def preprocess(document):
    # Tokenize the document into words
    tokens = word_tokenize(' '.join(document))
    # Lemmatize each word, convert to lowercase, and remove stop words and non-alphabetic tokens
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalpha() and token.lower() not in stop_words]
    return tokens

# Apply preprocessing to all documents
preprocessed_documents = [(preprocess(doc), category) for doc, category in documents]

# Print an example of a preprocessed document for verification
print("Sample preprocessed document:", preprocessed_documents[0][0][:10])
print("Label:", preprocessed_documents[0][1])

# Save preprocessed documents to a file for later use
with open('preprocessed_documents.pkl', 'wb') as f:
    pickle.dump(preprocessed_documents, f)
