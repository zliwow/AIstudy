import re
from collections import defaultdict
import math
import random

# Step 1: Preprocess the text
def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Remove numbers and punctuation
    text = re.sub(r'[^a-z\s]', '', text)
    # Tokenize the text into words
    words = text.split()
    return words

# Step 2: Count bigrams and unigrams
def count_ngrams(words):
    unigram_counts = defaultdict(int)
    bigram_counts = defaultdict(int)
    
    for i in range(len(words) - 1):
        unigram_counts[words[i]] += 1
        bigram_counts[(words[i], words[i+1])] += 1
    unigram_counts[words[-1]] += 1  # Count the last word
    
    return unigram_counts, bigram_counts

# Step 3: Calculate bigram probabilities
def calculate_bigram_probabilities(unigram_counts, bigram_counts):
    bigram_probs = {}
    for bigram, count in bigram_counts.items():
        bigram_probs[bigram] = count / unigram_counts[bigram[0]]
    return bigram_probs

# Step 4: Calculate perplexity for the relative frequencies model
def calculate_perplexity_relative_frequencies(bigram_probs, data, word_to_ix):
    total_log_prob = 0
    total_words = 0
    
    for context, target in data:
        context_word = [k for k, v in word_to_ix.items() if v == context][0]
        target_word = [k for k, v in word_to_ix.items() if v == target][0]
        bigram = (context_word, target_word)
        
        if bigram in bigram_probs:
            log_prob = math.log(bigram_probs[bigram])
        else:
            log_prob = -float('inf')  # If the bigram is not in the training set, assign a very low probability
        
        total_log_prob += log_prob
        total_words += 1
    
    avg_log_prob = total_log_prob / total_words
    perplexity = math.exp(-avg_log_prob)
    return perplexity

# Step 5: Generate text using the bigram probabilities with probabilities shown
def generate_text(bigram_probs, start_word, word_to_ix, ix_to_word, max_length=50):
    current_word = start_word
    generated_text = [current_word]
    
    for _ in range(max_length - 1):
        # Get possible next words and their probabilities
        next_words_probs = [(bigram[1], prob) for bigram, prob in bigram_probs.items() if bigram[0] == current_word]
        
        if not next_words_probs:
            break
        
        # Sort next words by their probabilities in descending order
        next_words_probs.sort(key=lambda x: x[1], reverse=True)
        
        # Show the probabilities of the top 5 possible next words
        print(f"Current word: {current_word}")
        top_next_words_probs = next_words_probs[:5]
        for next_word, prob in top_next_words_probs:
            print(f"Next word: {next_word}, Probability: {prob:.4f}")
        
        # Randomly select the next word from the top 5
        next_word = random.choice(top_next_words_probs)[0]
        generated_text.append(next_word)
        current_word = next_word

        print(f"Chosen next word: {next_word}\n")
    
    return ' '.join(generated_text)

# Main function to run the code
def main():
    # Load Warren Buffett's letters
    with open('WarrenBuffet.txt', 'r') as file:
        text = file.read()

    # Preprocess the text
    words = preprocess_text(text)

    # Build vocabulary and mappings
    vocab = list(set(words))
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}

    # Prepare training data
    data = [(word_to_ix[words[i]], word_to_ix[words[i+1]]) for i in range(len(words) - 1)]

    # Count unigrams and bigrams
    unigram_counts, bigram_counts = count_ngrams(words)

    # Calculate bigram probabilities
    bigram_probs = calculate_bigram_probabilities(unigram_counts, bigram_counts)

    # Generate text
    start_word = 'berkshire'  # Starting word for text generation
    generated_text = generate_text(bigram_probs, start_word, word_to_ix, ix_to_word, max_length=50)
    print(generated_text)

    # Calculate perplexity
    perplexity = calculate_perplexity_relative_frequencies(bigram_probs, data, word_to_ix)
    print(f"Relative Frequencies Model Perplexity: {perplexity}")

if __name__ == "__main__":
    main()



'''
When Always picking rank 1 prob

berkshire hathaway inc to the company has been a few years ago 
he had a few years ago he had a few years ago he had a few years ago 
he had a few years ago he had a few years ago he had a few years ago he had
'''

'''
When picking randomly from top 5 

berkshire shares are not all four segments of its a business that will be sure we are in a 
huge retroactive contracts in and a business value of a business we have been better yet known nor 
intuitive the company that we are likely be sure of its important role

Relative Frequencies Model Perplexity: 23.635004219864296
'''