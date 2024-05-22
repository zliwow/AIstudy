'''
Zihao Li
CS 6120
Assignment 1 Part 1 Step 2 - 3
'''

from nltk.corpus import gutenberg
from bpe import learn_bpe, encode, decode

# Uncomment to download necessary NLTK data
# nltk.download('gutenberg')

def preprocess_corpus(corpus):
    """
    Preprocess the corpus by separating characters with spaces and appending '</w>' to signify word boundaries.
    
    Args:
        corpus (str): The input text corpus.
    
    Returns:
        list: A list of preprocessed words.
    """
    return [' '.join(list(word)) + ' </w>' for word in corpus.split()]

def train_bpe_on_gutenberg():
    """
    Train the BPE model on selected books from the Gutenberg corpus.
    
    Returns:
        Counter: The final vocabulary after training.
        list: The list of merge operations performed.
    """
    # Load training books from Gutenberg corpus
    training_books = ['austen-emma.txt', 'blake-poems.txt', 'shakespeare-hamlet.txt']
    training_corpus = ' '.join([gutenberg.raw(book) for book in training_books])
    
    # Preprocess corpus
    preprocessed_corpus = preprocess_corpus(training_corpus)
    
    # Apply BPE
    num_merges = 100
    final_vocab, merge_operations, merge_frequencies = learn_bpe(preprocessed_corpus, num_merges)
    
    return final_vocab, merge_operations

def test_bpe_on_gutenberg(merge_operations):
    """
    Test the BPE model on selected books from the Gutenberg corpus by encoding and decoding the text.
    
    Args:
        merge_operations (list): The list of merge operations performed during training.
    
    Returns:
        tuple: A tuple containing the original text, encoded text, and decoded text.
    """
    # Load test books from Gutenberg corpus
    test_books = ['carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt']
    test_corpus = ' '.join([gutenberg.raw(book) for book in test_books])
    
    # Encode and decode using BPE
    encoded_text = encode(test_corpus, merge_operations)
    decoded_text = decode(encoded_text, merge_operations)
    
    return test_corpus, encoded_text, decoded_text

if __name__ == "__main__":
    final_vocab, merge_operations = train_bpe_on_gutenberg()
    original_text, encoded_text, decoded_text = test_bpe_on_gutenberg(merge_operations)
    
    print("Original Text Sample:", original_text[:500])
    print("Encoded Text Sample:", encoded_text[:500])
    print("Decoded Text Sample:", decoded_text[:500])
