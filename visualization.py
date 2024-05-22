'''
Zihao Li
CS 6120
Assignment 1 Part 1 Step 6
'''

import matplotlib.pyplot as plt
from tokenization_comparison import compare_tokenizations

def plot_vocab_comparison(initial_vocab_size, reference_vocab_size, bpe_vocab_size):
    """
    Plot a comparison of vocabulary sizes before and after applying BPE.
    
    Args:
        initial_vocab_size (int): The size of the initial vocabulary.
        reference_vocab_size (int): The size of the reference vocabulary using standard tokenization.
        bpe_vocab_size (int): The size of the vocabulary after applying BPE.
    """
    vocab_sizes = [initial_vocab_size, reference_vocab_size, bpe_vocab_size]
    labels = ['Initial Vocabulary', 'Reference Vocabulary', 'BPE Vocabulary']

    plt.figure(figsize=(10, 6))
    plt.bar(labels, vocab_sizes, color=['blue', 'orange', 'green'])
    plt.xlabel('Vocabulary')
    plt.ylabel('Size')
    plt.title('Vocabulary Size Comparison')
    plt.show()

if __name__ == "__main__":
    test_books = ['carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt']
    initial_vocab_size, reference_vocab_size, bpe_vocab_size = compare_tokenizations(test_books)
    print("Initial Vocabulary Size (before BPE):", initial_vocab_size)
    print("Reference Vocabulary Size:", reference_vocab_size)
    print("BPE Vocabulary Size:", bpe_vocab_size)
    plot_vocab_comparison(initial_vocab_size, reference_vocab_size, bpe_vocab_size)

