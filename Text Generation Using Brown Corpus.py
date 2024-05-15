import random
from nltk.corpus import brown

class CharNGramLanguageModel:
    def __init__(self, n):
        self.n = n
        self.char_counts = {}
        self.total_counts = {}

        for sentence in brown.sents():
            text = ' '.join(sentence) + ' <end-of-sequence>'
            for i in range(len(text) - n):
                char_sequence = text[i:i+n]
                next_char = text[i+n]

                if char_sequence not in self.char_counts:
                    self.char_counts[char_sequence] = {}
                    self.total_counts[char_sequence] = 0

                if next_char not in self.char_counts[char_sequence]:
                    self.char_counts[char_sequence][next_char] = 0

                self.char_counts[char_sequence][next_char] += 1
                self.total_counts[char_sequence] += 1

        for char_sequence in self.char_counts:
            for next_char in self.char_counts[char_sequence]:
                self.char_counts[char_sequence][next_char] /= self.total_counts[char_sequence]

    def generate_character(self, prompt):
        if len(prompt) < self.n:
            return random.choice(list(brown.words()))[:1]  # Return the first character of a random word

        last_chars = prompt[-self.n:]
        if last_chars not in self.char_counts:
            return None  # Return None if the last 'n' characters are not found in the training data

        next_char_probs = self.char_counts[last_chars]
        next_char = random.choices(list(next_char_probs.keys()), weights=list(next_char_probs.values()))[0]
        return next_char

    def generate(self, prompt):
        generated_text = prompt
        max_length = 100  # Set a maximum length to prevent infinite loops
        while len(generated_text) < max_length:
            next_char = self.generate_character(generated_text)
            if next_char is None or next_char == '<end-of-sequence>':
                break  # Stop generating if the next character is None or the end-of-sequence marker
            generated_text += next_char
        return generated_text

if __name__ == '__main__':
    n = int(input("Enter the value of n: "))
    model = CharNGramLanguageModel(n)
    prompt = input("Enter a prompt: ")
    generated_text = model.generate(prompt)
    print("Generated text:", generated_text)