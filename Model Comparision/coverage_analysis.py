
from collections import Counter
import matplotlib.pyplot as plt
import pickle

# Load preprocessed documents
with open('preprocessed_documents.pkl', 'rb') as f:
    preprocessed_documents = pickle.load(f)

# Flatten all preprocessed tokens into a single list
all_tokens = [token for doc, _ in preprocessed_documents for token in doc]

# Count the frequency of each token
token_counts = Counter(all_tokens)
total_unique_tokens = len(token_counts)  # Total number of unique tokens

# Function to calculate coverage
def calculate_coverage(token_counts, num_tokens):
    # Get the most common tokens up to num_tokens
    most_common_tokens = token_counts.most_common(num_tokens)
    # Calculate the number of covered tokens
    covered_tokens = sum(count for _, count in most_common_tokens)
    # Calculate total number of tokens
    total_tokens = sum(token_counts.values())
    # Return the coverage as a ratio
    return covered_tokens / total_tokens

# Calculate coverage at different token counts
token_counts_list = list(range(100, total_unique_tokens, 100))
coverage = [calculate_coverage(token_counts, num) for num in token_counts_list]

# Plot coverage analysis
plt.figure(figsize=(10, 6))
plt.plot(token_counts_list, coverage, marker='o')
plt.xlabel('Number of Tokens')
plt.ylabel('Coverage Percentage')
plt.title('Coverage Analysis')
plt.grid(True)
plt.show()
