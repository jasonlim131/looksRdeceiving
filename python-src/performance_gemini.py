from collections import Counter
import json
import os


MODEL = "gemini-1.5-flash"

def find_most_frequent(responses):
    frequency = {}
    for response in responses:
        if response in frequency:
            frequency[response] += 1
        else:
            frequency[response] = 1
    
    max_count = 0
    most_frequent = None
    for response, count in frequency.items():
        if count > max_count:
            max_count = count
            most_frequent = response
    
    return most_frequent

def calculate_distribution_and_score(data):
    distribution = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0
    }
    ground_truth_distribution = {
        "A": 0,
        "B": 0,
        "C": 0,
        "D": 0
    }
    total_responses = 0
    correct_responses = 0
    total_prompts = len(data['prompts'])

    for prompt_id, prompt_data in data['prompts'].items():
        correct_answer = prompt_data["correct_answer"]
        responses = prompt_data["responses"]
        
        # Update ground truth distribution
        if correct_answer in ground_truth_distribution:
            ground_truth_distribution[correct_answer] += 1
        
        for response in responses:
            # Clean up the response and extract the letter
            clean_response = response.strip().upper().replace(".", "")
            if clean_response in distribution:
                distribution[clean_response] += 1
                total_responses += 1
                if clean_response == correct_answer:
                    correct_responses += 1

    # Convert counts to percentages
    for key in distribution:
        distribution[key] = (distribution[key] / total_responses) * 100 if total_responses > 0 else 0

    # Convert ground truth counts to percentages
    for key in ground_truth_distribution:
        ground_truth_distribution[key] = (ground_truth_distribution[key] / total_prompts) * 100

    # Calculate overall score
    score = (correct_responses / total_responses) * 100 if total_responses > 0 else 0

    return distribution, ground_truth_distribution, score

def process_file(filepath, title, output_dir):
    with open(filepath, 'r') as f:
        data = json.load(f)

    distribution, ground_truth_distribution, vmmlu_score = calculate_distribution_and_score(data)

    os.makedirs(output_dir, exist_ok=True)

    # Remove "D" field from distributions if it's a Social IQA file
    if "Social_IQA" in title:
        distribution.pop("D", None)
        ground_truth_distribution.pop("D", None)
    
    # Save distributions to JSON
    distribution_json_path = os.path.join(output_dir, f"{title}_distribution.json")
    with open(distribution_json_path, mode='w') as file:
        json.dump({
            "model_distribution": distribution,
            "ground_truth_distribution": ground_truth_distribution
        }, file, indent=4)

    # Save vMMLU score to a text file
    score_path = os.path.join(output_dir, f"{title}_vmmlu_score.txt")
    with open(score_path, mode='w') as file:
        file.write(f"vMMLU Score: {vmmlu_score}")

    print(f"{title}:")
    print(f"Model Distribution: {distribution}")
    print(f"Ground Truth Distribution: {ground_truth_distribution}")
    print(f"vMMLU Score: {vmmlu_score}")
    print()

def main():

    files_to_process = {
        f"results/{MODEL}/results_neutral.json": "Neutral_vmmlu",
        f"results/{MODEL}/results_optionA.json": "OptionA_vmmlu",
        f"results/{MODEL}/results_optionB.json": "OptionB_vmmlu",
        f"results/{MODEL}/results_optionC.json": "OptionC_vmmlu",
        f"results/{MODEL}/results_optionD.json": "OptionD_vmmlu",
        f"results/v_social_i_qa/{MODEL}/results_neutral_blue_centered.json": "Neutral_Blue_Centered_Social_IQA",
        f"results/v_social_i_qa/{MODEL}/results_optionA_blue_centered.json": "OptionA_Blue_Centered_Social_IQA",
        f"results/v_social_i_qa/{MODEL}/results_optionB_blue_centered.json": "OptionB_Blue_Centered_Social_IQA",
        f"results/v_social_i_qa/{MODEL}/results_optionC_blue_centered.json": "OptionC_Blue_Centered_Social_IQA",
        f"results/v_social_i_qa/{MODEL}/results_neutral.json": "Neutral_Social_IQA",
        f"results/v_social_i_qa/{MODEL}/results_optionA.json": "OptionA_Social_IQA",
        f"results/v_social_i_qa/{MODEL}/results_optionB.json": "OptionB_Social_IQA",
        f"results/v_social_i_qa/{MODEL}/results_optionC.json": "OptionC_Social_IQA"
    }

    output_directory = f"distribution_and_score/{MODEL}"

    for filepath, title in files_to_process.items():
        if os.path.exists(filepath):
            process_file(filepath, title, output_directory)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main()