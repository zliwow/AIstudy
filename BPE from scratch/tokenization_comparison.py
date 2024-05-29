from nltk.corpus import gutenberg
from nltk.tokenize import word_tokenize, sent_tokenize
from train_test import  train_bpe_on_gutenberg
from bpe import encode

# Uncomment to download necessary NLTK data
# nltk.download('punkt')
# nltk.download('gutenberg')

def create_reference_tokenization(corpus):
    """
    Create a reference tokenization using NLTK's punkt tokenizer.
    
    Args:
        corpus (str): The input text corpus.
    
    Returns:
        list: A list of tokenized sentences, where each sentence is a list of words.
    """
    sentences = sent_tokenize(corpus)
    tokenized_corpus = [word_tokenize(sentence) for sentence in sentences]
    return tokenized_corpus

def save_reference_tokenization(file_path, tokenized_corpus):
    """
    Save the reference tokenization to a file.
    
    Args:
        file_path (str): The path to the file where the tokenization will be saved.
        tokenized_corpus (list): The tokenized corpus to save.
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        for sentence in tokenized_corpus:
            file.write(' '.join(sentence) + '\n')

def calculate_metrics(reference_tokenization, bpe_tokenization):
    """
    Calculate various metrics comparing the reference tokenization and BPE tokenization.
    
    Args:
        reference_tokenization (list): The reference tokenized corpus.
        bpe_tokenization (str): The BPE tokenized corpus.
    
    Returns:
        dict: A dictionary containing various comparison metrics.
    """
    reference_tokens = [token for sentence in reference_tokenization for token in sentence]
    bpe_tokens = bpe_tokenization.split()
    
    reference_vocab_size = len(set(reference_tokens))
    bpe_vocab_size = len(set(bpe_tokens))

    tokenization_accuracy = sum(1 for a, b in zip(reference_tokens, bpe_tokens) if a == b) / len(reference_tokens) * 100
    tokenization_coverage = len(set(bpe_tokens) & set(reference_tokens)) / len(set(reference_tokens)) * 100

    true_positives = len(set(reference_tokens) & set(bpe_tokens))
    false_positives = len(set(bpe_tokens) - set(reference_tokens))
    false_negatives = len(set(reference_tokens) - set(bpe_tokens))
    
    precision = true_positives / (true_positives + false_positives) if true_positives + false_positives > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if true_positives + false_negatives > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    jaccard_similarity = len(set(reference_tokens) & set(bpe_tokens)) / len(set(reference_tokens) | set(bpe_tokens))

    return {
        'reference_vocab_size': reference_vocab_size,
        'bpe_vocab_size': bpe_vocab_size,
        'tokenization_accuracy': tokenization_accuracy,
        'tokenization_coverage': tokenization_coverage,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'jaccard_similarity': jaccard_similarity
    }

def compare_tokenizations(test_books):
    """
    Compare the BPE tokenization with the reference tokenization for the given test books.
    
    Args:
        test_books (list): A list of book filenames to test.
    
    Returns:
        tuple: A tuple containing the initial vocabulary size, reference vocabulary size, and BPE vocabulary size.
    """
    initial_vocab, merge_operations = train_bpe_on_gutenberg()
    initial_vocab_size = len(initial_vocab)

    for book in test_books:
        # Load book
        raw_text = gutenberg.raw(book)
        
        # Create reference tokenization
        reference_tokenization = create_reference_tokenization(raw_text)
        
        # Save reference tokenization to a file
        reference_file_path = f'reference_tokenization_{book}.txt'
        save_reference_tokenization(reference_file_path, reference_tokenization)
        print(f"Reference tokenization saved to {reference_file_path}")
        
        # BPE tokenization
        encoded_text = encode(raw_text, merge_operations)
        
        # Calculate metrics
        metrics = calculate_metrics(reference_tokenization, encoded_text)
        
        # Print comparison
        print(f"\nComparison for {book}:")
        print(f"Initial Vocabulary Size (before BPE): {initial_vocab_size}")
        print(f"Reference Vocabulary Size: {metrics['reference_vocab_size']}")
        print(f"BPE Vocabulary Size: {metrics['bpe_vocab_size']}")
        print(f"Tokenization Accuracy: {metrics['tokenization_accuracy']:.2f}%")
        print(f"Tokenization Coverage: {metrics['tokenization_coverage']:.2f}%")
        print(f"Precision: {metrics['precision']:.2f}")
        print(f"Recall: {metrics['recall']:.2f}")
        print(f"F1-Score: {metrics['f1_score']:.2f}")
        print(f"Jaccard Similarity: {metrics['jaccard_similarity']:.2f}")

    return initial_vocab_size, metrics['reference_vocab_size'], metrics['bpe_vocab_size']

if __name__ == "__main__":
    test_books = ['carroll-alice.txt', 'chesterton-ball.txt', 'chesterton-brown.txt']
    initial_vocab_size, reference_vocab_size, bpe_vocab_size = compare_tokenizations(test_books)
    print("Initial Vocabulary Size (before BPE):", initial_vocab_size)
    print("Reference Vocabulary Size:", reference_vocab_size)
    print("BPE Vocabulary Size:", bpe_vocab_size)
