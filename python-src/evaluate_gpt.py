import os
import sys
import requests
import json
from typing import List, Dict
import base64
from typing import List, Tuple
import numpy as np  
from dotenv import load_dotenv
from datasets import load_dataset
import math
from openai import OpenAI
import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT
from tqdm import tqdm
import re
from collections import Counter
from itertools import chain


#load the environment variables file
load_dotenv()  # Load environment variables from .env file
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

#define endpoints
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
GOOGLE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"

MODEL = "gpt-4o-mini"

#load the tinyMMLU dataset
tiny_mmlu = load_dataset("tinyBenchmarks/tinyMMLU")
tiny_mmlu_test = tiny_mmlu['test']
input_formatted = tiny_mmlu_test['input_formatted']

bias_type = ["neutral", "optionA", "optionB", "optionC", "optionD"]



def get_prompt(input_formatted):
    '''
    Define prompt as a nested dictionary where:
    - The outer key is the input_formatted
    - The inner keys are the bias types
    - The values are the paths to the images in output_directory
    '''
    prompts = {}
    for i, question in enumerate(input_formatted):
        prompts[question] = {
            'neutral': f"output_directory/vmmlu_neutral_size_rendered/question_{i}.png",
            'optionA': f"output_directory/vmmlu_optionA_size_rendered/question_{i}.png",
            'optionB': f"output_directory/vmmlu_optionB_size_rendered/question_{i}.png",
            'optionC': f"output_directory/vmmlu_optionC_size_rendered/question_{i}.png",
            'optionD': f"output_directory/vmmlu_optionD_size_rendered/question_{i}.png"
        }
    return prompts

prompts = get_prompt(input_formatted)


def calculate_token_stats(all_token_logprobs, top_k=4):
    # Calculate average log probs
    print(len(all_token_logprobs))
    print("all token logprobs", all_token_logprobs[0])
    
    # Replace None values with 0
    all_token_logprobs = [{token: iteration.get(token, 0) for token in ['A', 'B', 'C', 'D']} for iteration in all_token_logprobs]
    
    avg_logprobs = {token: np.mean([iteration[token] for iteration in all_token_logprobs]) 
                    for token in ['A', 'B', 'C', 'D']}
    
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
    print("used image_path", image_path) 
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

    
def openai_query(prompt: str, image_path: str, model_name) -> str:
    ## Set the API key and model name
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>"))
    encoded_image = encode_image(image_path)
    
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer these questions to the best of your abilities"},  # <-- This is the system message that provides context to the model
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            } 
        ],
        temperature = 0,
        logprobs=True,
        top_logprobs=10
    )
    
    tokens_of_interest = ['A', 'B', 'C', 'D']
    
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

def openai_image_only_query(prompt: str, image_path: str, model_name) -> str:

    ## Set the API key and model name
    MODEL=model_name
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>"))
    encoded_image = encode_image(image_path)
    
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer these questions to the best of your abilities"},  # <-- This is the system message that provides context to the model
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{encoded_image}"
                        }
                    }
                ]
            } 
        ],
        temperature = 0,
        logprobs=True,
        top_logprobs=10
    )
    
    tokens_of_interest = ['A', 'B', 'C', 'D']
    
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


def openai_text_only_query(prompt: str, image_path: str, model_name) -> str:
    
    ## Set the API key and model name
    MODEL=model_name
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>"))
    encoded_image = encode_image(image_path)
    
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful assistant. Answer these questions to the best of your abilities"},  #system prompt
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_image}"}}
                ]
            } 
        ],
        temperature = 0,
        logprobs=True,
        top_logprobs=10
    )
    
    tokens_of_interest = ['A', 'B', 'C', 'D']
    
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



# regex to match the correct answer
def is_correct_answer(top_token, correct_answer):
    #print("top_token", top_token)
        
    # Rigorous String Matching
    if re.match(rf'^{correct_answer}\b', top_token):
        return True
    
    return False

def int_to_mcq_option(integer):
    return chr(65 + integer)


def run_variation_test(input_question, correct_answer, prompts, variation, model_name, num_iterations=10):
    logprob_list = []
    response_list = []
    correct_count = 0
    
    for i in tqdm(range(num_iterations)):
        query_response = openai_text_only_query(
            f"What is the correct answer? Answer only in the form of A, B, C, or D. {input_question}", 
            prompts[input_question][variation], 
            model_name
        )
        log_prob_dict = query_response[0]
        top_token = query_response[1]
        
        response_list.append(top_token)
        logprob_list.append(log_prob_dict)
                
        if is_correct_answer(top_token, correct_answer):
            correct_count += 1
    
    percent_correct = (correct_count / num_iterations) * 100
    
    #returns percent correct answer AND the list of dictionaries containing all logprobs for n iterations
    return percent_correct, response_list, logprob_list

# List of variations to test
variations = ["neutral", "optionA", "optionB", "optionC", "optionD"]

selected_prompts = list(range(100))
#selected_prompts = [prompt_index for prompt_index in selected_prompts if prompt_index not in [17, 18, 70, 94]]

# Initialize results dictionary
results = {variation: {"total_percent_correct": 0, "prompts": {}} for variation in variations}

# Load existing results if available to avoid overwriting progress
for variation in variations:
    try:
        with open(f"results/vmmlu/{MODEL}/results_{variation}_size.json", "r") as f:
            results[variation] = json.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, continue with an empty result set
        pass

# Main loop for processing prompts
for prompt_index in tqdm(selected_prompts, desc="Processing prompts"):
    input_prompt = input_formatted[prompt_index]
    short_question = tiny_mmlu_test['question'][prompt_index]
    correct_answer_int = tiny_mmlu_test['answer'][prompt_index]
    correct_answer = int_to_mcq_option(correct_answer_int)

    print(f"\nQuestion {prompt_index}:", short_question)
    print("\ncorrect_answer:", correct_answer)

    for variation in variations:
        print(f"\nRunning test for {variation} bias")
        
        # Check if the prompt has already been processed for the current variation
        if str(prompt_index) in results[variation]["prompts"]:
            print(f"Prompt {prompt_index} already processed for {variation} bias. Skipping.")
            continue
        
        percent_correct, responses, all_token_logprobs = run_variation_test(
            input_prompt, 
            correct_answer, 
            prompts, 
            variation, 
            MODEL
        )
        
        token_stats = calculate_token_stats(all_token_logprobs)
        
        results[variation]["total_percent_correct"] += percent_correct #this is wrong
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

        # Save results incrementally to avoid losing progress
        with open(f"results/vmmlu/{MODEL}/results_{variation}_size.json", "w") as f:
            json.dump(results[variation], f, indent=2)

        print(f"Percent Correct {variation} bias: {percent_correct}%")

# Calculate averages and store results
for variation in variations:
    avg_percent_correct = results[variation]["total_percent_correct"] / len(selected_prompts)
    results[variation]["avg_percent_correct"] = avg_percent_correct
    
    with open(f"results/vmmlu/{MODEL}/results_{variation}_size.json", "w") as f:
        json.dump(results[variation], f, indent=2)
    
    print(f"\nAverage Percent Correct for {variation} bias: {avg_percent_correct}%")
    print(f"Results for {variation} saved in results_{variation}.json")
    