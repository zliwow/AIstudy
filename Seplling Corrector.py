import numpy as np
import pandas as pd
from collections import defaultdict, Counter


# Load the dataset
file_path = '' # I erased my path for submission. 
with open(file_path, 'r') as file:
    data = file.readlines()

# Preprocess the data to get correct and incorrect spellings
corrections = {}
for line in data:
    parts = line.strip().split(': ')
    correct_word = parts[0].strip()
    incorrect_words = parts[1].strip().split()
    for word in incorrect_words:
        corrections[word] = correct_word

# Generate emission and transition probabilities
alphabet = 'abcdefghijklmnopqrstuvwxyz'

def generate_emission_probabilities(corrections):
    emission_counts = defaultdict(Counter)
    for incorrect, correct in corrections.items():
        for i in range(min(len(correct), len(incorrect))):
            c = correct[i]
            i = incorrect[i]
            emission_counts[c][i] += 1
    
    emission_probabilities = defaultdict(lambda: defaultdict(lambda: 1e-6))  # Smoothing
    for correct_letter, counter in emission_counts.items():
        total_count = sum(counter.values())
        for incorrect_letter, count in counter.items():
            emission_probabilities[correct_letter][incorrect_letter] = count / total_count
    
    return emission_probabilities

def generate_transition_probabilities(corrections):
    transition_counts = defaultdict(Counter)
    for correct in corrections.values():
        prev_char = '<s>'
        for char in correct:
            transition_counts[prev_char][char] += 1
            prev_char = char
        transition_counts[prev_char]['</s>'] += 1
    
    transition_probabilities = defaultdict(lambda: defaultdict(lambda: 1e-6))  # Smoothing
    for prev_char, counter in transition_counts.items():
        total_count = sum(counter.values())
        for current_char, count in counter.items():
            transition_probabilities[prev_char][current_char] = count / total_count
    
    return transition_probabilities

emission_probabilities = generate_emission_probabilities(corrections)
transition_probabilities = generate_transition_probabilities(corrections)

# Implement the Viterbi algorithm
def viterbi_algorithm(word, emission_probabilities, transition_probabilities):
    states = list(alphabet)
    V = [{}]
    
    # Initialization step
    for state in states:
        V[0][state] = {
            "prob": transition_probabilities['<s>'][state] * emission_probabilities[state][word[0]],
            "prev": None
        }
    
    # Recursion step
    for t in range(1, len(word)):
        V.append({})
        for state in states:
            max_tr_prob = max(V[t-1][prev_state]["prob"] * transition_probabilities[prev_state][state] for prev_state in states)
            for prev_state in states:
                if V[t-1][prev_state]["prob"] * transition_probabilities[prev_state][state] == max_tr_prob:
                    max_prob = max_tr_prob * emission_probabilities[state][word[t]]
                    V[t][state] = {"prob": max_prob, "prev": prev_state}
                    break
    
    # Termination step
    opt = []
    max_prob = max(value["prob"] for value in V[-1].values())
    previous = None
    for state, data in V[-1].items():
        if data["prob"] == max_prob:
            opt.append(state)
            previous = state
            break
    
    for t in range(len(V) - 2, -1, -1):
        opt.insert(0, V[t + 1][previous]["prev"])
        previous = V[t + 1][previous]["prev"]
    
    return ''.join(opt)

# Correct user input
def correct_text(input_text, emission_probabilities, transition_probabilities):
    words = input_text.split()
    corrected_words = []
    for word in words:
        corrected_word = viterbi_algorithm(word, emission_probabilities, transition_probabilities)
        corrected_words.append(corrected_word)
    return ' '.join(corrected_words)

# Check all incorrect spellings in the dataset
def check_all_corrections(corrections):
    results = []
    for incorrect, correct in corrections.items():
        corrected = viterbi_algorithm(incorrect, emission_probabilities, transition_probabilities)
        results.append((incorrect, corrected, correct, corrected == correct))
    return results

# Test the corrections
correction_results = check_all_corrections(corrections)

# Analyze the results for the three questions
def analyze_results_v4(correction_results):
    # Find a correctly spelled word incorrectly corrected by the algorithm
    first_example = None
    for correct in corrections.values():
        corrected = viterbi_algorithm(correct, emission_probabilities, transition_probabilities)
        if corrected != correct:
            first_example = (correct, corrected)
            break

    # Find an incorrectly spelled word incorrectly corrected by the algorithm
    second_example = None
    for incorrect, corrected, correct, is_correct in correction_results:
        if not is_correct:
            second_example = (incorrect, corrected, correct)
            break

    # Find an incorrectly spelled word correctly corrected by the algorithm
    third_example = None
    for incorrect, corrected, correct, is_correct in correction_results:
        if is_correct:
            third_example = (incorrect, corrected, correct)
            break

    return first_example, second_example, third_example

# Get the analysis results
first_example, second_example, third_example = analyze_results_v4(correction_results)

# Print the analysis results
print("Example of a word which was correctly spelled by the user, but incorrectly corrected by the algorithm:")
if first_example:
    print(f"  Correctly spelled: {first_example[0]}")
    print(f"  Corrected by algorithm: {first_example[1]}")
    print(f"  Correct spelling: {first_example[0]}")
else:
    print("  No example found.")

print("\nExample of a word which was incorrectly spelled by the user, but still incorrectly corrected by the algorithm:")
if second_example:
    print(f"  Incorrectly spelled: {second_example[0]}")
    print(f"  Corrected by algorithm: {second_example[1]}")
    print(f"  Correct spelling: {second_example[2]}")
else:
    print("  No example found.")

print("\nExample of a word which was incorrectly spelled by the user, and was correctly corrected by the algorithm:")
if third_example:
    print(f"  Incorrectly spelled: {third_example[0]}")
    print(f"  Corrected by algorithm: {third_example[1]}")
    print(f"  Correct spelling: {third_example[2]}")
else:
    print("  No example found.")
