import os
import sys
import requests
import json
from typing import List, Dict, Tuple
import base64
import numpy as np  
from dotenv import load_dotenv
from datasets import load_dataset
import math
import openai
from openai import OpenAI
from tqdm import tqdm
import re
from collections import Counter

# Load the environment variables file
# load_dotenv()  # Load environment variables from .env file
# OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
# ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
# GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

MODEL = "gpt-4o-mini"
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
print(OPENAI_API_KEY)

# Define endpoints
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
GOOGLE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
import json

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

def calculate_token_stats(all_token_logprobs, top_k=3):
    all_token_logprobs = [{token: iteration.get(token, 0) for token in ['A', 'B', 'C']} for iteration in all_token_logprobs]
    avg_logprobs = {token: np.mean([iteration[token] for iteration in all_token_logprobs]) for token in ['A', 'B', 'C']}
    sorted_tokens = sorted(avg_logprobs.items(), key=lambda x: x[1], reverse=True)
    top_k_tokens = sorted_tokens[:top_k]
    average_answer = top_k_tokens[0][0]
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



def openai_query(prompt, image_path, model_name):
    client = OpenAI(api_key=OPENAI_API_KEY)
    print("image path", image_path)
    encoded_image = encode_image(image_path)
    
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer these questions to the best of your abilities"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                ]
            }
        ],
        temperature=0,
        logprobs=True,
        top_logprobs=10
        
    )
    tokens_of_interest = ['A', 'B', 'C']
    
    top_token = completion.choices[0].message.content[0]
    logprob_content = completion.choices[0].logprobs.content[0].top_logprobs
    
    print("top_token", top_token)
    print("logprob_content", logprob_content)
    
    #take the 0th element of tuple
    logprob_dict = extract_token_logprobs(logprob_content, tokens_of_interest)[0] 

    print("Log probabilities for tokens of interest:")
    for token, logprob in logprob_dict.items():
        print(f"{token}: {logprob}")
    
    return logprob_dict, top_token

def run_variation_test(input_question, correct_answer, prompts, variation, model_name, num_iterations=10):
    input_question_index = selected_subset.index(input_question)
    print("input question index", input_question_index)
    logprob_list = []
    response_list = []
    correct_count = 0
    for i in tqdm(range(num_iterations)):
        query_response = openai_query(
            f"What is the correct answer? Answer only in the form of A, B, or C. {input_question}",
            prompts[input_question_index][variation],
            model_name
        )
        log_prob_dict, top_token = query_response
        response_list.append(top_token)
        logprob_list.append(log_prob_dict)
        if is_correct_answer(top_token, correct_answer):
            correct_count += 1
    percent_correct = (correct_count / num_iterations) * 100
    return percent_correct, response_list, logprob_list

def is_correct_answer(top_token, correct_answer):
    return top_token.strip().upper() == correct_answer.strip().upper()

def int_to_mcq_option(integer):
    print("integer", type(int(integer)))
    return chr(65 + int(integer) - 1)



# List of variations to test
variations = ["neutral", "optionA", "optionB", "optionC"]
selected_prompts = list(range(len(input_formatted)))
results = {variation: {"total_percent_correct": 0, "prompts": {}} for variation in variations}

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
        percent_correct, responses, all_token_logprobs = run_variation_test(
            input_prompt, 
            correct_answer, 
            prompts, 
            variation, 
            MODEL
        )
        token_stats = calculate_token_stats(all_token_logprobs)
        results[variation]["total_percent_correct"] += percent_correct
        results[variation]["prompts"][prompt_index] = {
            "input_prompt": input_prompt,
            "correct_answer": correct_answer,
            "percent_correct": percent_correct,
            "responses": responses,
            "token_logprobs": all_token_logprobs,
            "avg_logprobs": token_stats["avg_logprobs"],
            "top_k_tokens": token_stats["top_k_tokens"],
            "average_answer": token_stats["average_answer"],
            "most_frequent_response": token_stats["most_frequent_response"]
        }
        with open(f"results/v_social_i_qa/{MODEL}/results_{variation}_blue_centered.json", "w") as f:
            json.dump(results[variation], f, indent=2)

for variation in variations:
    avg_percent_correct = results[variation]["total_percent_correct"] / len(selected_prompts)
    results[variation]["avg_percent_correct"] = avg_percent_correct
    with open(f"results/v_social_i_qa/{MODEL}/results_{variation}_blue_centered.json", "w") as f:
        json.dump(results[variation], f, indent=2)
    print(f"\nAverage Percent Correct for {variation} bias: {avg_percent_correct}%")
    print(f"Results for {variation} saved in results_{variation}.json")