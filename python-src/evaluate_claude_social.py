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
import anthropic
from anthropic import HUMAN_PROMPT, AI_PROMPT
from tqdm import tqdm
import re
from collections import Counter
from itertools import chain
import PIL
import math


#load the environment variables file
load_dotenv()  # Load environment variables from .env file

# Get the value of ANTHROPIC_API_KEY from the environment
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
print(ANTHROPIC_API_KEY)

#define endpoints
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"

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

def get_image_prompt(input_formatted):
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

prompts = get_image_prompt(input_formatted)


prompts = get_image_prompt(input_formatted)



def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# given a string answer, find the first occurrence of A, B, C, or D
def find_first_answer(string):
    # Find all occurrences of A, B, C, or D that are not part of a word
    # Check for a non-alphabet character or end of string after A, B, C, or D
    matches = re.findall(r'\b([A-D])\b(?![a-zA-Z])', string)
    # Return the first occurrence, or None if not found
    return matches[0] if matches else None

    
model_names = {
    "sonnet-3.5": "claude-3-5-sonnet-20240620",
    "opus": "claude-3-opus-20240229",
    "sonnet3": "claude-3-sonnet-20240229",
    "haiku": "claude-3-haiku-20240307"
}

#define model globally
MODEL = model_names["haiku"]

def claude_query(prompt: str, image_path: str, model_name: str) -> Tuple[Dict[str, float], str]:
    # Encode the image
    
    
    encoded_image = encode_image(image_path)
    
    client = anthropic.Anthropic(
        # defaults to os.environ.get("ANTHROPIC_API_KEY")
        api_key=ANTHROPIC_API_KEY
    )

    completion = client.messages.create(
        model=MODEL,
        max_tokens=64,
        temperature=0.0,
        system="You are a helpful assistant. Answer these questions to the best of your abilities.",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": encoded_image
                        }
                    }
                ]
            }
        ],
        top_k=10
    )
    
    
    response = completion.content[0].text
    print("Top Token:", response)
    
    answer = find_first_answer(response)
    
    return answer
   

# regex to match the correct answer
def is_correct_answer(top_token, correct_answer):
    #print("top_token", top_token)
    if not isinstance(top_token, (str, bytes)):
        return False
    
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
        query_response = claude_query(
            f"Answer only in the form of A, B, or C. {input_question}.",
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
            with open(f"results/v_social_i_qa/{MODEL}/results_{variation}_blue_centered.json", "w") as f:
                json.dump(results[variation], f, indent=2)

    for variation in variations:
        avg_percent_correct = results[variation]["total_percent_correct"] / len(selected_prompts)
        results[variation]["avg_percent_correct"] = avg_percent_correct
        with open(f"results/v_social_i_qa/{MODEL}/results_{variation}_blue_centered.json", "w") as f:
            json.dump(results[variation], f, indent=2)
        print(f"\nAverage Percent Correct for {variation} bias: {avg_percent_correct}%")
        print(f"Results for {variation} saved in results_{variation}.json")
        
        
if __name__ == "__main__":
    main()