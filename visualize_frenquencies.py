'''
Zihao Li
CS 6120
Assignment 1 Part 1 Step 6
'''

import matplotlib.pyplot as plt
from bpe import learn_bpe
from train_test import preprocess_corpus

from nltk.corpus import gutenberg

def plot_merge_frequencies(merge_frequencies):
    """
    Plot the frequency of byte pair merges during the BPE process.
    
    Args:
        merge_frequencies (list): A list of frequencies for each merge operation.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(merge_frequencies) + 1), merge_frequencies, marker='o')
    plt.xlabel('Merge Step')
    plt.ylabel('Frequency')
    plt.title('Frequency of Byte Pair Merges')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # Load training books from Gutenberg corpus
    training_books = ['austen-emma.txt', 'blake-poems.txt', 'shakespeare-hamlet.txt']
    training_corpus = ' '.join([gutenberg.raw(book) for book in training_books])
    
    # Preprocess corpus
    preprocessed_corpus = preprocess_corpus(training_corpus)
    
    # Learn BPE and get merge frequencies
    num_merges = 100
    final_vocab, merge_operations, merge_frequencies = learn_bpe(preprocessed_corpus, num_merges)
    
    # Plot merge frequencies
    plot_merge_frequencies(merge_frequencies)
