import numpy as np
import json
from scipy import stats
from sklearn.metrics import roc_curve, auc
from typing import Dict, List, Tuple
import os
import sys

# Set the root directory of the project
root_directory = os.path.dirname(os.path.abspath(__file__))
root_directory = os.path.abspath(os.path.join(root_directory, "../"))
os.chdir(root_directory)
# Set the system path to the same directory
sys.path.append(root_directory)
print("Root directory set to:", root_directory)

def percent_correct(data):
    correct_count_frequent = 0
    correct_count_average = 0
    total_count = 0

    for prompt_num, prompt in data['prompts'].items():
        correct_answer = prompt['correct_answer']
        most_frequent_response = prompt['most_frequent_response']
        average_response = prompt['average_answer']

        if correct_answer == most_frequent_response:
            correct_count_frequent += 1
        if correct_answer == average_response:
            correct_count_average += 1
        total_count += 1

    percent_correct_frequent = (correct_count_frequent / total_count) * 100
    percent_correct_average = (correct_count_average / total_count) * 100

    return percent_correct_frequent, percent_correct_average

def count_answer_distribution(data):
    distribution = {'A': 0, 'B': 0, 'C': 0, 'D': 0}

    for prompt_num, prompt in data['prompts'].items():
        correct_answer = prompt['correct_answer']
        distribution[correct_answer] += 1

    return distribution
#load data for variations
def load_data_for_variations(file_path: str) -> List[Dict]:
    variations = ['neutral', 'optionA', 'optionB', 'optionC', 'optionD']
    data = []
    for variation in variations:
        variation_file_path = f"/Users/crayhippo/vmmlu/results/results_{variation}.json"
        with open(variation_file_path, 'r') as f:
            prompt_data = json.load(f)
            data.append(prompt_data)
    return data

def calculate_linear_probability(logprob: float) -> float:
    return np.exp(logprob)

    

def top_token_changed(neutral: Dict, biased: Dict) -> bool:
    return max(neutral, key=neutral.get) != max(biased, key=biased.get)

def is_biased_option_new_top(neutral: Dict, biased: Dict, biased_option: str) -> bool:
    neutral_top = max(neutral, key=neutral.get)
    biased_top = max(biased, key=biased.get)
    return neutral_top != biased_top and biased_top == biased_option

def calculate_probability_increase(neutral: Dict, biased: Dict, option: str) -> float:
    neutral_prob = calculate_linear_probability(neutral[option])
    biased_prob = calculate_linear_probability(biased[option])
    return (biased_prob - neutral_prob) / neutral_prob

def analyze_top_token_changes(data: List[Dict]) -> Dict[str, int]:
    changes = {bias_type: 0 for bias_type in ['optionA', 'optionB', 'optionC', 'optionD']}
    
    for d in data:
        print("prompt keys", d.keys())
        print("changes keys", changes.keys())
        answers = d
        print("neutral", answers)
        for bias_type in changes.keys():
            biased = d['token_logprobs'][0]
            if is_biased_option_new_top(neutral, biased, bias_type[-1]):
                changes[bias_type] += 1
    return changes

def analyze_probability_increases(data: List[Dict], threshold: float = 0.1) -> Dict[str, List[Tuple[int, float]]]:
    significant_increases = {bias_type: [] for bias_type in ['optionA', 'optionB', 'optionC', 'optionD']}
    for i, prompt in enumerate(data):
        neutral = prompt['neutral']['token_logprobs'][0]
        for bias_type in significant_increases.keys():
            biased = prompt[bias_type]['token_logprobs'][0]
            option = bias_type[-1]
            increase = calculate_probability_increase(neutral, biased, option)
            if increase > threshold:
                significant_increases[bias_type].append((i, increase))
    return significant_increases

def calculate_roc(data: List[Dict]) -> Dict[str, Tuple[float, float, float]]:
    roc_results = {}
    for bias_type in ['optionA', 'optionB', 'optionC', 'optionD']:
        y_true = []
        y_scores = []
        for prompt in data:
            correct_answer = prompt['correct_answer']
            biased_prob = calculate_linear_probability(prompt[bias_type]['token_logprobs'][0][bias_type[-1]])
            y_true.append(1 if correct_answer == bias_type[-1] else 0)
            y_scores.append(biased_prob)
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        roc_results[bias_type] = (fpr, tpr, roc_auc)
    return roc_results

def identify_interesting_prompts(data: List[Dict], threshold: float = 0.5) -> List[int]:
    interesting_prompts = []
    for i, prompt in enumerate(data):
        neutral = prompt['neutral']['token_logprobs'][0]
        for bias_type in ['optionA', 'optionB', 'optionC', 'optionD']:
            biased = prompt[bias_type]['token_logprobs'][0]
            option = bias_type[-1]
            increase = calculate_probability_increase(neutral, biased, option)
            if abs(increase) > threshold or is_biased_option_new_top(neutral, biased, option):
                interesting_prompts.append(i)
                break
    return interesting_prompts

def analyze_bias_effectiveness_by_logprob_range(data: List[Dict], num_bins: int = 10) -> Dict[str, List[Tuple[float, float, float]]]:
    """
    Analyze the effectiveness of bias across different ranges of initial log probabilities.
    
    Args:
    data (List[Dict]): The dataset containing prompts and their variations.
    num_bins (int): The number of log probability bins to create.
    
    Returns:
    Dict[str, List[Tuple[float, float, float]]]: For each bias type, a list of tuples containing
    (logprob_bin_start, average_probability_increase, number_of_samples) for each bin.
    """
    bias_effectiveness = {bias_type: [] for bias_type in ['optionA', 'optionB', 'optionC', 'optionD']}
    
    for bias_type in bias_effectiveness.keys():
        logprobs = []
        prob_increases = []
        
        # Collect logprobs and probability increases
        for prompt in data:
            neutral = prompt['neutral']['token_logprobs'][0]
            biased = prompt[bias_type]['token_logprobs'][0]
            option = bias_type[-1]
            
            neutral_logprob = neutral[option]
            logprobs.append(neutral_logprob)
            
            increase = calculate_probability_increase(neutral, biased, option)
            prob_increases.append(increase)
        
        # Create bins
        logprob_bins = np.linspace(min(logprobs), max(logprobs), num_bins + 1)
        
        # Analyze effectiveness for each bin
        for i in range(num_bins):
            bin_start = logprob_bins[i]
            bin_end = logprob_bins[i + 1]
            
            # Find probability increases for logprobs in this bin
            bin_increases = [inc for lp, inc in zip(logprobs, prob_increases) if bin_start <= lp < bin_end]
            
            if bin_increases:
                avg_increase = np.mean(bin_increases)
                bias_effectiveness[bias_type].append((bin_start, avg_increase, len(bin_increases)))
    
    return bias_effectiveness


def main():
    
    variations = load_data_for_variations("results")    
    text_only = json.load(open("results/results_text_only_neutral.json"))
    image_only = json.load(open("results/results_image_only_neutral.json"))
        
    for prompt_num, prompt in enumerate(variations):
        print(f"percent correct for {prompt_num}", percent_correct(prompt))
    
    print(f"percent correct for text only(most_frequent, avg_answer): {percent_correct(text_only)}")
    print("answer distribution:", count_answer_distribution(text_only))
    
    # # Analyze top token changes
    # top_token_changes = analyze_top_token_changes(data)
    # print("Top token changes:", top_token_changes)
    
    # # Analyze significant probability increases
    # prob_increases = analyze_probability_increases(data)
    # print("Significant probability increases:", prob_increases)
    
    # # Calculate ROC curves
    # roc_results = calculate_roc(data)
    # for bias_type, (fpr, tpr, roc_auc) in roc_results.items():
    #     print(f"ROC AUC for {bias_type}: {roc_auc}")
    
    # # Identify interesting prompts
    # interesting_prompts = identify_interesting_prompts(data)
    # print("Interesting prompts for case study:", interesting_prompts)
    
    # # Analyze bias effectiveness by logprob range
    # bias_effectiveness = analyze_bias_effectiveness_by_logprob_range(data)
    
    # for bias_type, effectiveness in bias_effectiveness.items():
    #     print(f"\nBias effectiveness for {bias_type}:")
    #     for logprob_start, avg_increase, num_samples in effectiveness:
    #         print(f"  LogProb range starting at {logprob_start:.2f}: "
    #               f"Avg probability increase: {avg_increase:.4f}, "
    #               f"Number of samples: {num_samples}")

if __name__ == "__main__":
    main()