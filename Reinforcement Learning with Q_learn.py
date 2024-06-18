import random
import numpy as np
from collections import defaultdict
from scipy.special import softmax

class CharNGramLanguageModel:
    """
    A character-level n-gram language model.
    
    Attributes:
        n (int): The length of the n-grams.
        model (defaultdict): A nested dictionary to store the weights of next characters given an n-gram.
        characters (str): A string containing all possible characters.
    """

    def __init__(self, n=3):
        """
        Initializes the CharNGramLanguageModel with the specified n-gram length.

        Args:
            n (int): The length of the n-grams. Default is 3.
        """
        self.n = n
        self.model = defaultdict(lambda: defaultdict(float))
        self.characters = 'abcdefghijklmnopqrstuvwxyz '

    def update_weights(self, ngram, next_char, reward, alpha, gamma, next_max_q):
        """
        Updates the weights for the given n-gram and next character using Q-learning.

        Args:
            ngram (str): The current n-gram.
            next_char (str): The next character.
            reward (float): The reward received.
            alpha (float): The learning rate.
            gamma (float): The discount factor.
            next_max_q (float): The maximum Q-value of the next state.
        """
        old_weight = self.model[ngram][next_char]
        new_weight = old_weight + alpha * (reward + gamma * next_max_q - old_weight)
        self.model[ngram][next_char] = new_weight

    def normalize_weights(self):
        """
        Normalizes the weights using the softmax function to convert them to probabilities.
        """
        for ngram, next_chars in self.model.items():
            weights = np.array(list(next_chars.values()))
            if len(weights) == 0:
                continue
            normalized_weights = softmax(weights)
            for i, char in enumerate(next_chars):
                next_chars[char] = normalized_weights[i]

    def generate_text(self, seed, length=100):
        """
        Generates text of the specified length using the current model starting with the given seed.

        Args:
            seed (str): The initial seed text to start the generation.
            length (int): The length of the text to generate. Default is 100 characters.

        Returns:
            str: The generated text.
        """
        current_seq = seed[-self.n:]
        generated = seed
        for _ in range(length):
            next_char_probs = self.model[current_seq]
            if not next_char_probs:
                next_char = random.choice(self.characters)
            else:
                next_char = random.choices(
                    list(next_char_probs.keys()), 
                    list(next_char_probs.values())
                )[0]
            generated += next_char
            current_seq = generated[-self.n:]
        return generated

class ReinforcementLearning:
    """
    A class to perform Q-learning on a CharNGramLanguageModel.

    Attributes:
        model (CharNGramLanguageModel): The character-level n-gram language model.
        alpha (float): The learning rate.
        gamma (float): The discount factor.
    """

    def __init__(self, model, alpha=0.1, gamma=0.9):
        """
        Initializes the ReinforcementLearning with the specified model, learning rate, and discount factor.

        Args:
            model (CharNGramLanguageModel): The character-level n-gram language model.
            alpha (float): The learning rate. Default is 0.1.
            gamma (float): The discount factor. Default is 0.9.
        """
        self.model = model
        self.alpha = alpha
        self.gamma = gamma

    def Q_learn(self, criteria, prompt, iterations_per_prompt=30):
        """
        Performs Q-learning to update the model weights based on the specified criteria.

        Args:
            criteria (function): A function that takes a string and returns a numerical score as the reward.
            prompt (str): The initial prompt to start the generation.
            iterations_per_prompt (int): The number of iterations to perform for each prompt. Default is 30.
        """
        for _ in range(iterations_per_prompt):
            ngram = prompt[-self.model.n:]
            next_char = random.choice(self.model.characters)
            generated_text = prompt + next_char
            reward = criteria(generated_text)
            next_char_values = list(self.model.model[ngram].values())
            next_max_q = max(next_char_values) if next_char_values else 0
            self.model.update_weights(ngram, next_char, reward, self.alpha, self.gamma, next_max_q)
            prompt += next_char

def word_criteria(text):
    """
    Criteria function that rewards shorter text.

    Args:
        text (str): The generated text.

    Returns:
        float: The negative length of the text.
    """
    return -len(text)

def main():
    """
    Main function to test the CharNGramLanguageModel and ReinforcementLearning classes.
    """
    with open("lines.txt", "r", encoding="utf-8") as file:
        text = file.read().replace("\n", " ")

    prompts = ["Water", "Earth", "Fire", "Air", "Long Agp", "The", "Four", "Nations", "Lived", "Together", "In", "Harmony"]

    for prompt in prompts:
        print(f"Testing with prompt: {prompt}")
        ngram_model = CharNGramLanguageModel(n=3)

        # Generate initial texts
        initial_texts = [ngram_model.generate_text(seed=prompt, length=100) for _ in range(10)]
        initial_average_length = np.mean([len(word) for text in initial_texts for word in text.split()])
        print(f"Initial average length: {initial_average_length}")
        print(f"Sample text: {initial_texts[0]}")

        # Perform Q-learning
        rl = ReinforcementLearning(model=ngram_model)
        rl.Q_learn(criteria=word_criteria, prompt=prompt, iterations_per_prompt=30)

        # Normalize weights after Q-learning
        ngram_model.normalize_weights()

        # Generate final texts
        final_texts = [ngram_model.generate_text(seed=prompt, length=100) for _ in range(10)]
        final_average_length = np.mean([len(word) for text in final_texts for word in text.split()])
        print(f"Final average length: {final_average_length}")
        print(f"Sample text: {final_texts[0]}")
        print("=" * 80)

if __name__ == "__main__":
    main()
