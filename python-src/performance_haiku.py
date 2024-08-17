import json
import os

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
    correct_count = 0
    total_count = 0

    for prompt_id, prompt_data in data['prompts'].items():
        correct_answer = prompt_data["correct_answer"]
        responses = prompt_data["responses"]
        total_count += len(responses)

        # Update ground truth distribution
        if correct_answer in ground_truth_distribution:
            ground_truth_distribution[correct_answer] += 1

        for response in responses:
            if response is None or response not in distribution:
                continue
            distribution[response] += 1
            if response == correct_answer:
                correct_count += 1

    # Convert counts to percentages
    for key in distribution:
        distribution[key] = (distribution[key] / total_count) * 100

    # Convert ground truth counts to percentages
    total_ground_truth = sum(ground_truth_distribution.values())
    for key in ground_truth_distribution:
        ground_truth_distribution[key] = (ground_truth_distribution[key] / total_ground_truth) * 100

    vmmlu_score = int((correct_count / total_count) * 100)

    return distribution, ground_truth_distribution, vmmlu_score

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
    # Your files_to_process dictionary remains the same
    files_to_process = {
        "results/claude-3-haiku-20240307/results_neutral.json": "Neutral_Haiku",
        "results/claude-3-haiku-20240307/results_optionA.json": "OptionA_Haiku",
        "results/claude-3-haiku-20240307/results_optionB.json": "OptionB_Haiku",
        "results/claude-3-haiku-20240307/results_optionC.json": "OptionC_Haiku",
        "results/claude-3-haiku-20240307/results_optionD.json": "OptionD_Haiku",
        "results/v_social_i_qa/claude-3-haiku-20240307/results_neutral_blue_centered.json": "Neutral_Blue_Centered_Social_IQA",
        "results/v_social_i_qa/claude-3-haiku-20240307/results_optionA_blue_centered.json": "OptionA_Blue_Centered_Social_IQA",
        "results/v_social_i_qa/claude-3-haiku-20240307/results_optionB_blue_centered.json": "OptionB_Blue_Centered_Social_IQA",
        "results/v_social_i_qa/claude-3-haiku-20240307/results_optionC_blue_centered.json": "OptionC_Blue_Centered_Social_IQA",
        "results/v_social_i_qa/claude-3-haiku-20240307/results_neutral.json": "Neutral_Social_IQA",
        "results/v_social_i_qa/claude-3-haiku-20240307/results_optionA.json": "OptionA_Social_IQA",
        "results/v_social_i_qa/claude-3-haiku-20240307/results_optionB.json": "OptionB_Social_IQA",
        "results/v_social_i_qa/claude-3-haiku-20240307/results_optionC.json": "OptionC_Social_IQA"
    }
    MODEL = "claude-3-haiku-20240307"
    output_directory = f"distribution_and_score/{MODEL}"

    for filepath, title in files_to_process.items():
        if os.path.exists(filepath):
            process_file(filepath, title, output_directory)
        else:
            print(f"File not found: {filepath}")

if __name__ == "__main__":
    main()