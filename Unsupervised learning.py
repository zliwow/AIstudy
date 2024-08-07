import pandas as pd
import string
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

# Load the datasets
df_fake = pd.read_csv(r'C:\Users\Andrew\Desktop\NEU\5100 foundation for AI\HW9\Fake.csv')
df_real = pd.read_csv(r'C:\Users\Andrew\Desktop\NEU\5100 foundation for AI\HW9\True.csv')

# Add a label column to distinguish between real and fake news
df_fake['label'] = 'fake'
df_real['label'] = 'real'

# Combine the datasets
df = pd.concat([df_fake, df_real], ignore_index=True)

# Data Preprocessing
def preprocess(text):
    text = text.lower()  # Convert to lowercase
    text = text.translate(str.maketrans('', '', string.punctuation))  # Remove punctuation
    text = ' '.join([word for word in text.split() if word not in ENGLISH_STOP_WORDS])  # Remove stopwords
    return text

df['processed_text'] = df['text'].apply(preprocess)

print(df.head()) 

# Vectorize the processed text
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['processed_text'])

# Fit LDA model with 10 topics
lda = LatentDirichletAllocation(n_components=10, random_state=0)
lda.fit(X)

def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print(f"Topic #{topic_idx}:")
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))
    print()

n_top_words = 10  # You can choose 10 or 20 as suggested
print_top_words(lda, vectorizer.get_feature_names_out(), n_top_words)

'''
Topic #0:
said percent government party new million year election reuters state
Topic #1:
tax trump wall january border merkel news president donald realdonaldtrump
Topic #2:
said senate house republican republicans president congress court obama law
Topic #3:
said people reuters government military security refugees united country state
Topic #4:
trump said clinton donald president campaign republican hillary presidential election
Topic #5:
said clinton department fbi trump intelligence investigation russian president information
Topic #6:
said trump united president states north korea china reuters nuclear
Topic #7:
people like just black america don white know video right
Topic #8:
said state syria saudi syrian islamic israel government iraq military
Topic #9:
said police court people law case man told family just

I think the topics generated by the LDA model represent real-world topics adaquetly, with focus on politics, international relations, etc.
'''

# Select random samples of real and fake news
real_news_sample = df[df['label'] == 'real'].sample(5, random_state=0)
fake_news_sample = df[df['label'] == 'fake'].sample(5, random_state=0)

print("Real News Sample:")
print(real_news_sample['processed_text'])
print("\nFake News Sample:")
print(fake_news_sample['processed_text'])

# Transform the text data to LDA topic distributions
real_topic_dist = lda.transform(vectorizer.transform(real_news_sample['processed_text']))
fake_topic_dist = lda.transform(vectorizer.transform(fake_news_sample['processed_text']))

print("Real News Topic Distributions:")
print(real_topic_dist)
print("\nFake News Topic Distributions:")
print(fake_topic_dist)

'''
Seems like real news documents are primarily related to international relations and politics.
Fake news focus on controversies on trump cascaded issues, social problems and board issues. 

The 5 news are draw at random, everytime it could be different. 
'''

# Transform the text data to LDA topic distributions
X_lda = lda.transform(vectorizer.transform(df['processed_text']))

# Convert labels to binary values: real -> 1, fake -> 0
y = df['label'].map({'real': 1, 'fake': 0})

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_lda, y, test_size=0.2, random_state=0)

# Train the Logistic Regression classifier
clf = LogisticRegression(random_state=0)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Examine coefficients
coefficients = clf.coef_[0]
print(f"Coefficients: {coefficients}")

# Map coefficients to topics
topic_names = [f'Topic #{i}' for i in range(len(coefficients))]
coeff_df = pd.DataFrame({'Topic': topic_names, 'Coefficient': coefficients})
coeff_df = coeff_df.sort_values(by='Coefficient', ascending=False)

print(coeff_df)

'''
Accuracy: 0.8710467706013363
Coefficients: [  3.5467719   -2.55087481   3.75493091   3.6852319   -0.95546623
   0.71812711   3.33415071 -13.73923099   3.20784971  -1.33164977]
      Topic  Coefficient
2  Topic #2     3.754931
3  Topic #3     3.685232
0  Topic #0     3.546772
6  Topic #6     3.334151
8  Topic #8     3.207850
5  Topic #5     0.718127
4  Topic #4    -0.955466
9  Topic #9    -1.331650
1  Topic #1    -2.550875
7  Topic #7   -13.739231

Topics such as Topic #2 (highly positive), Topic #3, and Topic #0 are most useful in predicting real news, while Topic #7 (highly negative) and Topic #1 are most indicative of fake news.
'''

# Filter the dataset to include only real news
real_news = df[df['label'] == 'real']

# Transform the text data to LDA topic distributions
X_real_lda = lda.transform(vectorizer.transform(real_news['processed_text']))

# Apply KMeans clustering with K=10
kmeans = KMeans(n_clusters=10, random_state=0)
real_news['cluster'] = kmeans.fit_predict(X_real_lda)

# Select 5 news documents from each cluster
for cluster_num in range(10):
    print(f"Cluster {cluster_num}:")
    cluster_sample = real_news[real_news['cluster'] == cluster_num].sample(5, random_state=0)
    for idx, doc in enumerate(cluster_sample['processed_text']):
        print(f"Document {idx+1}: {doc[:500]}...")  # Print the first 500 characters for brevity
    print("\n")


'''
Document 1: rome reuters pope francis implicitly criticized united states monday pulling paris agreement climate change praising means control devastating ...
Document 2: mexico city reuters mexican president enrique pena nieto meet republican presidential candidate donald trump private meeting wednesday pena nieto’s office ...
Document 3: zurich reuters companies blame sanctions stopping investing iran state department official told businesses wednesday saying risks putting wouldbe investors ...
Document 4: moscow reuters russia s defence ministry saturday criticized german defence minister ursula von der leyen saying bewildered assertion moscow planned send 100000 troops ...
Document 5: reuters highlights day president donald trump’s administration sunday intelligence official rejects trump’s accusation ...

This is shortened version of the printed result, i think in general the clusters of real news documents correspond to distinct themes from the csv file.
'''