'''
Zihao Li
CS 6120
Assignment 1 Part 1 Step 1
'''

from collections import Counter, defaultdict
import re

def get_initial_vocab(corpus):
    """
    Create the initial vocabulary from the corpus by counting the frequency of each character.
    
    Args:
        corpus (str): The input text corpus.
    
    Returns:
        Counter: A counter object with character frequencies.
    """
    vocab = Counter(corpus.replace(" ", ""))
    return vocab

def get_pairs(vocab):
    """
    Identify all pairs of consecutive characters in the vocabulary and count their frequency.
    
    Args:
        vocab (Counter): A counter object with character frequencies.
    
    Returns:
        defaultdict: A dictionary with pairs of characters and their frequencies.
    """
    pairs = defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i + 1]] += freq
    return pairs

def merge_vocab(pair, vocab):
    """
    Merge the most frequent pair of characters into a new token and update the vocabulary.
    
    Args:
        pair (tuple): The pair of characters to merge.
        vocab (Counter): The vocabulary counter object.
    
    Returns:
        Counter: The updated vocabulary after merging the pair.
    """
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)')
    new_vocab = {}
    for word in vocab:
        new_word = p.sub(''.join(pair), word)
        new_vocab[new_word] = vocab[word]
    return new_vocab

def learn_bpe(corpus, num_merges):
    """
    Learn the BPE merge operations by iteratively merging the most frequent pairs of characters.
    
    Args:
        corpus (list): A list of preprocessed words in the corpus.
        num_merges (int): The number of merge operations to perform.
    
    Returns:
        Counter: The final vocabulary after all merge operations.
        list: A list of merge operations performed.
        list: A list of frequencies of the merge operations.
    """
    vocab = Counter(corpus)
    merge_operations = []
    merge_frequencies = []

    for _ in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        vocab = merge_vocab(best, vocab)
        merge_operations.append(best)
        merge_frequencies.append(pairs[best])
    
    return vocab, merge_operations, merge_frequencies

def encode(text, merge_operations):
    """
    Encode the text using the learned BPE merge operations.
    
    Args:
        text (str): The input text to encode.
        merge_operations (list): The list of merge operations.
    
    Returns:
        str: The encoded text.
    """
    words = text.split()
    for pair in merge_operations:
        pattern = re.escape(' '.join(pair))
        p = re.compile(pattern)
        words = [p.sub(''.join(pair), word) for word in words]
    return ' '.join(words)

def decode(encoded_text, merge_operations):
    """
    Decode the encoded text back to its original form using the merge operations.
    
    Args:
        encoded_text (str): The encoded text to decode.
        merge_operations (list): The list of merge operations.
    
    Returns:
        str: The decoded text.
    """
    words = encoded_text.split()
    for pair in reversed(merge_operations):
        pattern = ''.join(pair)
        p = re.compile(pattern)
        words = [p.sub(' '.join(pair), word) for word in words]
    return ' '.join(words)