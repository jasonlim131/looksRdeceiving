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

sys.path.append('/Users/crayhippo/vmmlu')
os.chdir('/Users/crayhippo/vmmlu')

#load the environment variables file
load_dotenv()  # Load environment variables from .env file

# Get the value of ANTHROPIC_API_KEY from the environment
ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
print(ANTHROPIC_API_KEY)

#define endpoints
ANTHROPIC_ENDPOINT = "https://api.anthropic.com/v1/messages"
GOOGLE_ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"


#load the tinyMMLU dataset
tiny_mmlu = load_dataset("tinyBenchmarks/tinyMMLU")
tiny_mmlu_test = tiny_mmlu['test']
input_formatted = tiny_mmlu_test['input_formatted']

bias_type = ["neutral", "optionA", "optionB", "optionC", "optionD"]




def get_image_prompt(input_formatted):
    '''
    Define prompt as a nested dictionary where:
    - The outer key is the input_formatted
    - The inner keys are the bias types
    - The values are the paths to the images in output_directory
    '''
    prompts = {}
    for i, question in enumerate(input_formatted):
        prompts[question] = {
            'neutral': f"output_directory/vmmlu_neutral_rendered/question_{i}.png",
            'optionA': f"output_directory/vmmlu_optionA_rendered/question_{i}.png",
            'optionB': f"output_directory/vmmlu_optionB_rendered/question_{i}.png",
            'optionC': f"output_directory/vmmlu_optionC_rendered/question_{i}.png",
            'optionD': f"output_directory/vmmlu_optionD_rendered/question_{i}.png"
        }
    return prompts

prompts = get_image_prompt(input_formatted)


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
MODEL = model_names["sonnet3"]

def claude_query(prompt: str, image_path: str, model_name: str) -> Tuple[Dict[str, float], str]:
    # Set the API key and model name
    
    print("model", MODEL)

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
    

def int_to_mcq_option(integer):
    return chr(65 + integer)


def run_variation_test(input_question, correct_answer, prompts, variation, model_name, num_iterations=10):
    response_list = []
    correct_count = 0
    print("used image path", prompts[input_question][variation])
    
    for i in tqdm(range(num_iterations)):
        query_response = claude_query(
            f"USER: Answer only in the form of A, B, C, or D. {input_question}", 
            prompts[input_question][variation], 
            model_name
        )
        top_token = query_response
        
        response_list.append(top_token)
                
        if is_correct_answer(top_token, correct_answer):
            correct_count += 1
    
    percent_correct = (correct_count / num_iterations) * 100
    
    #returns percent correct answer AND the list of dictionaries containing all logprobs for n iterations
    return percent_correct, response_list

# List of variations to test
variations = ["neutral", "optionA", "optionB", "optionC", "optionD"]

# 0 to 99 in tiny mmlu
selected_prompts = list(range(100))

# Initialize results dictionary
results = {variation: {"total_percent_correct": 0, "prompts": {}} for variation in variations}

def main(): 
    
    # Load existing results if available to avoid overwriting progress
    for variation in variations:
        try:
            with open(f"results/vmmlu/{MODEL}/results_{variation}.json", "r") as f:
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
            
            percent_correct, responses = run_variation_test(
                input_prompt, 
                correct_answer, 
                prompts, 
                variation, 
                MODEL #change model name here
            )
                        
            results[variation]["total_percent_correct"] += percent_correct #this is wrong
            results[variation]["prompts"][prompt_index] = {
                "input_prompt": input_prompt,
                "correct_answer": correct_answer,
                "percent_correct": percent_correct,
                "responses": responses
            }

            # Save results incrementally intto avoid losing progress
            with open(f"results/vmmlu/{MODEL}/results_{variation}.json", "w") as f:
                json.dump(results[variation], f, indent=2)

            print(f"Percent Correct {variation} bias: {percent_correct}%")

    # Calculate averages and store results
    for variation in variations:
        avg_percent_correct = results[variation]["total_percent_correct"] / len(selected_prompts)
        results[variation]["avg_percent_correct"] = avg_percent_correct
        
        #######
        with open(f"results/vmmlu/{MODEL}/results_{variation}.json", "w") as f:
            json.dump(results[variation], f, indent=2)
        
        print(f"\nAverage Percent Correct for {variation} bias: {avg_percent_correct}%")
        print(f"Results for {variation} saved in results_{variation}.json")
        
        
if __name__ == "__main__":
    main()