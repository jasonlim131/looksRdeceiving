import os
import sys
import requests
import json
from typing import List, Dict
import base64
from typing import List, Tuple
import numpy as np  
import PIL
from dotenv import load_dotenv
from datasets import load_dataset
import math
from tqdm import tqdm
import re
from collections import Counter
from itertools import chain
import google.generativeai as genai


# specify whether the prompt is centered or not
cenetered = False

## for each dataset, change lines 36-40, 57-60, 181, 219, and  


#load the environment variables file
load_dotenv()  # Load environment variables from .env file
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)
print("API Key:", GOOGLE_API_KEY)

#define endpoints
GOOGLE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

# Load the selected subset
with open("output_directory/v_social_i_qa_info/selected_subset_seed_2(blue).json", "r") as f:
    selected_subset = json.load(f)

# Load the correct answers
with open("output_directory/v_social_i_qa_info/correct_answers_seed_2(blue).txt", "r") as f:
    correct_answers = f.read().splitlines()

# Prepare the dataset
input_formatted = [(entry) for entry in selected_subset]
print("input formatted", input_formatted[0])
print("length of input_formatted", len(input_formatted))
print("length of answers", len(correct_answers))

bias_type = ["neutral", "optionA", "optionB", "optionC"]

MODEL = "gemini-1.5-flash"

def get_prompt(input_formatted):
    prompts = {}
    for i, _ in enumerate(input_formatted):
        prompts[i] = {
            'neutral': f"output_directory/v_social_i_qa_centered_blue_neutral/question_{i}.png",
            'optionA': f"output_directory/v_social_i_qa_centered_blue_optionA/question_{i}.png",
            'optionB': f"output_directory/v_social_i_qa_centered_blue_optionB/question_{i}.png",
            'optionC': f"output_directory/v_social_i_qa_centered_blue_optionC/question_{i}.png"
        }
    # print("prompts", prompts[0]['optionA'])
    # print("length of prompts", len(prompts))
    return prompts

prompts = get_prompt(input_formatted)


def calculate_token_stats(all_token_logprobs, top_k=4):
    # Calculate average log probs
    print(len(all_token_logprobs))
    print("all token logprobs", all_token_logprobs[0])
    
    # Replace None values with 0
    all_token_logprobs = [{token: iteration.get(token, 0) for token in ['A', 'B', 'C']} for iteration in all_token_logprobs]
    
    avg_logprobs = {token: np.mean([iteration[token] for iteration in all_token_logprobs]) 
                    for token in ['A', 'B', 'C']}
    
    # Sort tokens by average log prob
    sorted_tokens = sorted(avg_logprobs.items(), key=lambda x: x[1], reverse=True)
    
    # Get top k tokens
    top_k_tokens = sorted_tokens[:top_k]
    
    # Determine the average answer (highest average log prob)
    average_answer = top_k_tokens[0][0]
    
    # Count most frequent response
    all_responses = [max(iteration, key=iteration.get) for iteration in all_token_logprobs]
    most_frequent_response = Counter(all_responses).most_common(1)[0][0]
    
    return {
        "avg_logprobs": avg_logprobs,
        "top_k_tokens": dict(top_k_tokens),
        "average_answer": average_answer,
        "most_frequent_response": most_frequent_response
    }

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


# helper method to extract the most likely token and logprobs for tokens of interest
def extract_token_logprobs(logprobs_content: List, tokens_of_interest: set) -> Tuple[Dict[str, float], str]:
    result = {
        logprob.token: round(logprob.logprob, 3)
        for logprob in logprobs_content
        if logprob.token in tokens_of_interest
    }
    
    # Fill in any missing tokens with negative infinity
    result.update({token: -math.inf for token in tokens_of_interest if token not in result})
    
    # Find the token with the highest logprob
    highest_logprob_token = max(logprobs_content, key=lambda x: x.logprob).token

    return result, highest_logprob_token

def gemini_query_safe(prompt: str, image_path: str, model_name: str) -> str:
    model = genai.GenerativeModel(model_name = model_name)
    img = PIL.Image.open(image_path)
    try:
        response = model.generate_content([img, prompt])
        return response.text
    
    # skip prompts that cause exceptions
    except Exception as e:
        print(f"Prompt causing exception: {prompt} - {str(e)}")
        return None

# regex to match the correct answer
def is_correct_answer(top_token, correct_answer):
        
    # Rigorous String Matching
    if re.match(rf'^{correct_answer}\b', top_token):
        return True
    
    return False

def int_to_mcq_option(charred_int):
    return chr(65 + int(charred_int) - 1)



def run_variation_test(input_question, correct_answer, prompts, variation, model_name, num_iterations=10):
    input_question_index = selected_subset.index(input_question)
    response_list = []
    correct_count = 0
    print("used image path", prompts[input_question_index][variation])
    
    for i in tqdm(range(num_iterations)):
        query_response = gemini_query_safe(
            f"Answer only in the form of A, B, or C. {input_question}",
            prompts[input_question_index][variation],
            model_name
        )
        if query_response is None:
            print(f"Skipping banned prompt: {input_question}")
            continue
        top_token = query_response
        response_list.append(top_token)
        if is_correct_answer(top_token, correct_answer):
            correct_count += 1
    percent_correct = (correct_count / num_iterations) * 100
    return percent_correct, response_list

# List of variations to test
variations = ["neutral", "optionA", "optionB", "optionC"]

# 0 to 99 in tiny mmlu
selected_prompts = list(range(100))
# Initialize results dictionary
results = {variation: {"total_percent_correct": 0, "prompts": {}} for variation in variations}

def main(): 
    #Load existing results if available to avoid overwriting progress
    for variation in variations:
        try:
            with open(f"results/v_social_i_qa/{MODEL}/results_{variation}_blue_centered.json", "r") as f:
                results[variation] = json.load(f)
        except FileNotFoundError:
            # If the file doesn't exist, continue with an empty result set
            pass

    for prompt_index in tqdm(selected_prompts, desc="Processing prompts"):
        input_prompt = input_formatted[prompt_index]
        short_question = selected_subset[prompt_index]['question']
        correct_answer = int_to_mcq_option(correct_answers[prompt_index])

        for variation in variations:
            if str(prompt_index) in results[variation]["prompts"]:
                continue
            percent_correct, responses = run_variation_test(
                input_prompt, 
                correct_answer, 
                prompts, 
                variation, 
                MODEL
            )
            
            results[variation]["total_percent_correct"] += percent_correct
            results[variation]["prompts"][prompt_index] = {
                "input_prompt": input_prompt,
                "correct_answer": correct_answer,
                "percent_correct": percent_correct,
                "responses": responses
            }
            # results[variation]["prompts"][prompt_index] = {
            #     "input_prompt": input_prompt,
            #     "correct_answer": correct_answer,
            #     "percent_correct": percent_correct,
            #     "responses": responses,
                
            #     "average_answer": token_stats["average_answer"],
            #     "most_frequent_response": token_stats["most_frequent_response"]
            # }
            with open(f"results/v_social_i_qa/{MODEL}/results_{variation}_blue_centered.json", "w") as f:
                json.dump(results[variation], f, indent=2)

    for variation in variations:
        avg_percent_correct = results[variation]["total_percent_correct"] / len(selected_prompts)
        results[variation]["avg_percent_correct"] = avg_percent_correct
        with open(f"results/v_social_i_qa/{MODEL}/results_{variation}_blue_centered.json", "w") as f:
            json.dump(results[variation], f, indent=2)
        print(f"\nAverage Percent Correct for {variation} bias: {avg_percent_correct}%")        
        
if __name__ == "__main__":
    main()