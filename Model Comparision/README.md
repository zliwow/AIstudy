Requirements
- Python 
- NLTK
- Scikit-learn
- Matplotlib

Necessary NLTK datasets:

import nltk
nltk.download('movie_reviews')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

Steps to Run:

python data_preparation.py
python coverage_analysis.py (will show converage result, but the png is also in the zipfile)
python classifier_and_visualization.py (will show visualization result, but the png is also in the zipfile)

Project Structure:
data_preparation.py: Handles data loading, preprocessing (tokenization, lemmatization, stop word removal), and saving the preprocessed data.

coverage_analysis.py: Performs coverage analysis on the preprocessed data, calculating and plotting the coverage percentage for different token counts.

classifier_and_visualization.py: Implements and evaluates Naive Bayes, Logistic Regression, and MLP classifiers with both TF and TF-IDF features, and visualizes the results.