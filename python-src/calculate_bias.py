import numpy as np
import json
from typing import Dict, List

MODEL = "llava"

def load_data_for_variations(file_path: str) -> List[Dict]:
    variations = ['neutral', 'optionA', 'optionB', 'optionC', 'optionD']
    data = []
    for variation in variations:
        variation_file_path = f"{file_path}/results_{MODEL}_{variation}.json"
        with open(variation_file_path, 'r') as f:
            prompt_data = json.load(f)
            data.append(prompt_data)
    return data

def percent_correct(data):
    correct_count = 0
    total_count = 0

    for prompt_num, prompt in data['prompts'].items():
        correct_answer = prompt['correct_answer']
        model_answer = prompt['most_frequent_response']

        if correct_answer == model_answer:
            correct_count += 1
        total_count += 1

    percent_correct = (correct_count / total_count) * 100
    return percent_correct

def count_answer_distribution(data):
    distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

    for prompt_num, prompt in data['prompts'].items():
        model_answer = prompt['most_frequent_response']
        distribution[model_answer] += 1

    return distribution

def analyze_answer_switches(neutral_data, biased_data):
    switches = {'total': 0, 'to_biased': 0}

    for prompt_num, neutral_prompt in neutral_data['prompts'].items():
        biased_prompt = biased_data['prompts'][prompt_num]
        neutral_answer = neutral_prompt['most_frequent_response']
        biased_answer = biased_prompt['most_frequent_response']
        correct_answer = neutral_prompt['correct_answer']

        if neutral_answer != biased_answer:
            switches['total'] += 1
            if biased_answer == biased_data['bias_option']:
                switches['to_biased'] += 1

    switches['percent_to_biased'] = (switches['to_biased'] / switches['total']) * 100 if switches['total'] > 0 else 0
    return switches

import numpy as np

def analyze_logprob_shifts(neutral_data, biased_data):
    def logprob_to_linear(logprob):
        return np.exp(logprob)

    options = ['A', 'B', 'C', 'D'] if 'D' in neutral_data['prompts'][next(iter(neutral_data['prompts']))]['log_probs'][0] else ['A', 'B', 'C']
    
    results = {}
    for option in options:
        deltas = []
        changes = []
        
        for prompt_id in neutral_data['prompts']:
            neutral_logprobs = [neutral_data['prompts'][prompt_id]['log_probs'][i][option] for i in range(10)]
            biased_logprobs = [biased_data['prompts'][prompt_id]['log_probs'][i][option] for i in range(10)]
            
            # Exclude iterations with -inf log probabilities
            neutral_logprobs = [lp for lp in neutral_logprobs if lp != float('-inf')]
            biased_logprobs = [lp for lp in biased_logprobs if lp != float('-inf')]
            
            if not neutral_logprobs or not biased_logprobs:
                continue
            
            # Calculate average logprob for this prompt
            neutral_mean = np.mean(neutral_logprobs)
            biased_mean = np.mean(biased_logprobs)
            
            # Convert logprob difference to linear probability
            delta_linear = logprob_to_linear(biased_mean) - logprob_to_linear(neutral_mean)
            deltas.append(delta_linear)
            
            # Calculate changes for variance
            neutral_linear = [logprob_to_linear(lp) for lp in neutral_logprobs]
            biased_linear = [logprob_to_linear(lp) for lp in biased_logprobs]
            changes.extend(np.array(biased_linear) - np.array(neutral_linear))

        # Average delta across prompts
        results[f'delta_{option}'] = np.mean(deltas)
        
        # Calculate variance of changes
        results[f'variance_change_{option}'] = np.var(changes)

    return results


def main():
    file_name = f"results/{MODEL}"
    variations = load_data_for_variations(file_name)
    
    for variation_num, variation in enumerate(variations[1:], start=1):  # Skip neutral, start from biased variations
        variation_char = chr(ord('A') + variation_num - 1)
        print(f"Variation {variation_char}:")
        
        logprob_shifts = analyze_logprob_shifts(variations[0], variation)
        print("Logprob shifts:")
        for key, value in logprob_shifts.items():
            print(f"  {key}: {value:.6f}")
        
        print()


if __name__ == "__main__":
    main()