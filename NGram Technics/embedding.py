import re
import torch
import torch.nn as nn
import torch.optim as optim
import random
import math

# Step 1: Preprocess the text
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    return words

# Step 2: Create vocabulary and word to index mapping
def build_vocab(words):
    vocab = list(set(words))
    word_to_ix = {word: i for i, word in enumerate(vocab)}
    ix_to_word = {i: word for i, word in enumerate(vocab)}
    return vocab, word_to_ix, ix_to_word

# Step 3: Prepare data for training
def prepare_data(words, word_to_ix):
    data = [(word_to_ix[words[i]], word_to_ix[words[i+1]]) for i in range(len(words) - 1)]
    return data

# Step 4: Define the model
class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super(BigramLanguageModel, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        embeds = self.embeddings(inputs)
        out = torch.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = torch.log_softmax(out, dim=1)
        return log_probs

# Step 5: Train the model
def train_model(model, data, epochs=50, lr=0.001):
    loss_fn = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for epoch in range(epochs):
        total_loss = 0
        for context, target in data:
            context_var = torch.tensor([context], dtype=torch.long)
            target_var = torch.tensor([target], dtype=torch.long)

            model.zero_grad()
            log_probs = model(context_var)
            loss = loss_fn(log_probs, target_var)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        if epoch % 1 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")

# Step 6: Generate text
def generate_text(model, word_to_ix, ix_to_word, start_word, max_length=50):
    current_word = start_word
    generated_text = [current_word]

    for _ in range(max_length - 1):
        current_word_idx = torch.tensor([word_to_ix[current_word]], dtype=torch.long)
        with torch.no_grad():
            log_probs = model(current_word_idx)
        top5_probs, top5_idx = torch.topk(log_probs, 5)
        
        top5_words = [ix_to_word[idx.item()] for idx in top5_idx[0]]
        print(f"Current word: {current_word}")
        for word, prob in zip(top5_words, top5_probs[0]):
            print(f"Next word: {word}, Log Probability: {prob.item():.4f}")

        next_word_idx = random.choice(top5_idx[0]).item()
        next_word = ix_to_word[next_word_idx]
        
        generated_text.append(next_word)
        current_word = next_word

        print(f"Chosen next word: {next_word}\n")

    return ' '.join(generated_text)

# Step 7: Calculate perplexity for the embeddings model
def calculate_perplexity_embeddings(model, data):
    total_loss = 0
    total_words = 0
    loss_fn = nn.NLLLoss()
    
    for context, target in data:
        context_var = torch.tensor([context], dtype=torch.long)
        target_var = torch.tensor([target], dtype=torch.long)
        
        with torch.no_grad():
            log_probs = model(context_var)
        
        loss = loss_fn(log_probs, target_var)
        total_loss += loss.item()
        total_words += 1
    
    avg_loss = total_loss / total_words
    perplexity = math.exp(avg_loss)
    return perplexity

# Main function to run the code
def main():
    # Load Warren Buffett's letters
    with open('WarrenBuffet.txt', 'r') as file:
        text = file.read()

    # Preprocess the text
    words = preprocess_text(text)[:2000]

    # Build vocabulary and mappings
    vocab, word_to_ix, ix_to_word = build_vocab(words)

    # Prepare training data
    data = prepare_data(words, word_to_ix)

    # Initialize model parameters
    embedding_dim = 100
    vocab_size = len(vocab)

    # Create the model
    model = BigramLanguageModel(vocab_size, embedding_dim)

    # Train the model
    train_model(model, data, epochs=50, lr=0.001)

    # Generate text
    start_word = 'berkshire'  # Starting word for text generation
    generated_text = generate_text(model, word_to_ix, ix_to_word, start_word, max_length=50)
    print(generated_text)

    # Calculate perplexity
    perplexity = calculate_perplexity_embeddings(model, data)
    print(f"Embeddings Model Perplexity: {perplexity}")

if __name__ == "__main__":
    main()

'''
Parameters need further optimization

dim 100, epochs 50, at first 2000

berkshire hathaway would be teaming to be the us with their and in 
investments compound growth for a year geico s and in this figure to a private our gain is in this 
emphasis he is since been run their earnings large and more berkshire is in this regard 

Embeddings Model Perplexity: 17.450820372477732

dim 50 epoch 50, entire file

berkshire weekend have the journalists to pm to buy and i am to be sure thing i will 
again set new berkshire and other current liabilities with this group to pm with their efforts go the 
journalists with their cooperation the most cases and the annual meeting and also present

Embeddings Model Perplexity: 60.01785757282546

dim 50 epoch 5, entire file

berkshire to pm of these a number to the future of the journalists of 
our annual of a huge the future for berkshire to the future of our insurance operations of 
our business we have to pm on a number and also and a will again have been of these
Embeddings Model Perplexity: 465.50712740562835

'''