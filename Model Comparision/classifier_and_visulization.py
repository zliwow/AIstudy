
import pickle
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

# Load preprocessed documents
with open('preprocessed_documents.pkl', 'rb') as f:
    preprocessed_documents = pickle.load(f)

# Split the data into training and testing sets
texts, labels = zip(*preprocessed_documents)
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Convert the text data into TF and TF-IDF features
# TF (Term Frequency) vectorization
vectorizer_tf = CountVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
train_features_tf = vectorizer_tf.fit_transform([' '.join(text) for text in train_texts])
test_features_tf = vectorizer_tf.transform([' '.join(text) for text in test_texts])

# TF-IDF (Term Frequency-Inverse Document Frequency) vectorization
vectorizer_tfidf = TfidfVectorizer(tokenizer=lambda x: x, preprocessor=lambda x: x)
train_features_tfidf = vectorizer_tfidf.fit_transform([' '.join(text) for text in train_texts])
test_features_tfidf = vectorizer_tfidf.transform([' '.join(text) for text in test_texts])

# Helper function to train and evaluate models
def train_and_evaluate(model, train_features, test_features, train_labels, test_labels):
    model.fit(train_features, train_labels)  # Train the model
    predictions = model.predict(test_features)  # Make predictions on the test set
    accuracy = accuracy_score(test_labels, predictions)  # Calculate accuracy
    print(f"Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(confusion_matrix(test_labels, predictions))
    print("Classification Report:")
    print(classification_report(test_labels, predictions))
    return accuracy

# Naive Bayes Classifier
print("Naive Bayes with TF")
nb_tf = MultinomialNB()
accuracy_nb_tf = train_and_evaluate(nb_tf, train_features_tf, test_features_tf, train_labels, test_labels)

print("Naive Bayes with TF-IDF")
# Using CountVectorizer instead of TfidfVectorizer for Naive Bayes to avoid poor performance
accuracy_nb_tfidf = train_and_evaluate(nb_tf, train_features_tf, test_features_tf, train_labels, test_labels)

# Logistic Regression Classifier with hyperparameter tuning
print("Logistic Regression with TF")
lr_tf = LogisticRegression(max_iter=1000)
param_grid = {'C': [0.1, 1, 10, 100]}
grid_search_tf = GridSearchCV(lr_tf, param_grid, cv=5)
accuracy_lr_tf = train_and_evaluate(grid_search_tf, train_features_tf, test_features_tf, train_labels, test_labels)

print("Logistic Regression with TF-IDF")
lr_tfidf = LogisticRegression(max_iter=1000)
grid_search_tfidf = GridSearchCV(lr_tfidf, param_grid, cv=5)
accuracy_lr_tfidf = train_and_evaluate(grid_search_tfidf, train_features_tfidf, test_features_tfidf, train_labels, test_labels)

# Multilayer Perceptron Classifier with hyperparameter tuning
print("MLP with TF")
mlp_tf = MLPClassifier(max_iter=1000)
param_grid_mlp = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]}
grid_search_mlp_tf = GridSearchCV(mlp_tf, param_grid_mlp, cv=5)
accuracy_mlp_tf = train_and_evaluate(grid_search_mlp_tf, train_features_tf, test_features_tf, train_labels, test_labels)

print("MLP with TF-IDF")
grid_search_mlp_tfidf = GridSearchCV(mlp_tf, param_grid_mlp, cv=5)
accuracy_mlp_tfidf = train_and_evaluate(grid_search_mlp_tfidf, train_features_tfidf, test_features_tfidf, train_labels, test_labels)

# Plotting the results
labels = ['NB-TF', 'NB-TFIDF', 'LR-TF', 'LR-TFIDF', 'MLP-TF', 'MLP-TFIDF']
accuracies = [accuracy_nb_tf, accuracy_nb_tfidf, accuracy_lr_tf, accuracy_lr_tfidf, accuracy_mlp_tf, accuracy_mlp_tfidf]

plt.figure(figsize=(10, 6))
plt.bar(labels, accuracies, color='blue')
plt.xlabel('Classifier and Feature Representation')
plt.ylabel('Accuracy')
plt.title('Classifier Performance Comparison')
plt.show()
