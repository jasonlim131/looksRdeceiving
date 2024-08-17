import os
import json
import random
from PIL import Image, ImageDraw, ImageFont
import textwrap
from tqdm import tqdm
from playwright.sync_api import sync_playwright

def load_social_iqa_data(data_file, labels_file, num_samples=100):
    with open(data_file, 'r') as f:
        data = [json.loads(line) for line in f]
    
    with open(labels_file, 'r') as f:
        labels = [int(line.strip()) for line in f]
    
    # Ensure data and labels have the same length
    assert len(data) == len(labels), "Mismatch between data and labels length"
    
    # Create a list of tuples (data, label)
    combined = list(zip(data, labels))
    
    # Randomly sample from the combined list
    sampled = random.sample(combined, num_samples)
    
    # Unzip the sampled data and labels
    sampled_data, sampled_labels = zip(*sampled)
    
    return list(sampled_data), list(sampled_labels)


def generate_html(item, variation):
    template = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Social IQA Question</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                background-color: white;
                color: black;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
            }}
            .container {{
                width: 80%;
                max-width: 600px;
            }}
            h2 {{
                margin-bottom: 20px;
            }}
            .option {{
                display: flex;
                align-items: center;
                margin: 10px 0;
            }}
            .radio {{
                width: 20px;
                height: 20px;
                border-radius: 50%;
                border: 2px solid black;
                margin-right: 10px;
                display: flex;
                justify-content: center;
                align-items: center;
            }}
            .text {{
                flex: 1;
            }}
            .highlight {{
                background-color: #b3d7fe;
                color: black;
                padding: 2px 5px;
                border-radius: 3px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h2>{context}</h2>
            <p><strong>{question}</strong></p>
            <div class="option">
                <div class="radio"></div>
                <div class="text">{answer_a}</div>
            </div>
            <div class="option">
                <div class="radio"></div>
                <div class="text">{answer_b}</div>
            </div>
            <div class="option">
                <div class="radio"></div>
                <div class="text">{answer_c}</div>
            </div>
        </div>
    </body>
    </html>
    """
    
    answers = {
        'neutral': [item['answerA'], item['answerB'], item['answerC']],
        'optionA': [f'<span class="highlight">{item["answerA"]}</span>', item['answerB'], item['answerC']],
        'optionB': [item['answerA'], f'<span class="highlight">{item["answerB"]}</span>', item['answerC']],
        'optionC': [item['answerA'], item['answerB'], f'<span class="highlight">{item["answerC"]}</span>']
    }
    
    return template.format(
        context=item['context'],
        question=item['question'],
        answer_a=answers[variation][0],
        answer_b=answers[variation][1],
        answer_c=answers[variation][2]
    )

def html_to_png(html_content, output_path, width=800, height=600):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.set_content(html_content)
        page.set_viewport_size({"width": width, "height": height})
        page.screenshot(path=output_path)
        browser.close()

def generate_visual_social_iqa(data, output_dir):
    variations = ['neutral', 'optionA', 'optionB', 'optionC']
    
    for variation in variations:
        os.makedirs(os.path.join(output_dir, f'v_social_i_qa_centered_blue_{variation}'), exist_ok=True)
    
    for idx, item in enumerate(tqdm(data)):
        for variation in variations:
            html_content = generate_html(item, variation)
            png_path = os.path.join(output_dir, f'v_social_i_qa_centered_blue_{variation}', f"question_{idx}.png")
            html_to_png(html_content, png_path)

def save_subset_info(data, labels, output_dir):
    with open(os.path.join(output_dir, 'selected_subset_blue_centered.json'), 'w') as f:
        json.dump(data, f, indent=2)
    
    with open(os.path.join(output_dir, 'correct_answers_blue_centered.txt'), 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

def main():
    output_dir = 'output_directory'
    os.makedirs(output_dir, exist_ok=True)
    
    data, labels = load_social_iqa_data('dev.jsonl', 'dev-labels.lst', num_samples=100)
    
    generate_visual_social_iqa(data, output_dir)
    save_subset_info(data, labels, output_dir)

if __name__ == "__main__":
    main()