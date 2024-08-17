import os
import json
import random
from tqdm import tqdm
from playwright.sync_api import sync_playwright

def load_social_iqa_data(file_path, num_samples=100):
    def load_labels(file_path):
        with open(file_path, 'r') as f:
            return [int(line.strip()) for line in f]
    
    labels = load_labels('dev-labels.lst')
    
    with open(file_path, 'r') as f:
        data = [json.loads(line) for line in f]
    original_indices = random.sample(range(len(data)), num_samples)
    # returns quesiton and answer pair
    return [(data[idx], labels[idx]) for idx in original_indices]




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
                background-color: #2c3e50;
                color: white;
                padding: 20px;
            }}
            .question {{
                font-size: 16px;
                margin-bottom: 20px;
            }}
            .choices {{
                display: flex;
                flex-direction: column;
            }}
            .choice {{
                margin: 10px 0;
                padding: 10px;
                background-color: #34495e;
                border-radius: 5px;
                display: flex;
                align-items: center;
            }}
            .bubble {{
                width: 20px;
                height: 20px;
                border-radius: 50%;
                border: 2px solid white;
                margin-right: 10px;
                display: inline-block;
            }}
            .selected {
            background-color: #f1c40f;
            color: #2c3e50;
            }}
            .selected .bubble {{
                background-color: #3498db;
                border-color: #3498db;
            }}
        </style>
    </head>
    <body>
        <div class="question">{context}</div>
        <div class="question"><strong>{question}</strong></div>
        <div class="choices">
            <div class="choice{selected_a}">
                <div class="bubble"></div>
                <span>{answer_a}</span>
            </div>
            <div class="choice{selected_b}">
                <div class="bubble"></div>
                <span>{answer_b}</span>
            </div>
            <div class="choice{selected_c}">
                <div class="bubble"></div>
                <span>{answer_c}</span>
            </div>
        </div>
    </body>
    </html>
    """
    
    selected = {
        'neutral': ['', '', ''],
        'optionA': [' selected', '', ''],
        'optionB': ['', ' selected', ''],
        'optionC': ['', '', ' selected']
    }
    print("item context", item.keys())
    
    return template.format(
        context=item['context'],
        question=item['question'],
        answer_a=item['answerA'],
        answer_b=item['answerB'],
        answer_c=item['answerC'],
        selected_a=selected[variation][0],
        selected_b=selected[variation][1],
        selected_c=selected[variation][2]
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
        os.makedirs(os.path.join(output_dir, f'v_social_i_qa_{variation}'), exist_ok=True)
    
    for idx, item in enumerate(tqdm(data)):
        for variation in variations:
            html_content = generate_html(item, variation)
            png_path = os.path.join(output_dir, f'v_social_i_qa_{variation}', f"question_{idx}.png")
            html_to_png(html_content, png_path)

def save_subset_info(data, labels, output_dir):
    with open(os.path.join(output_dir, 'selected_subset.json'), 'w') as f:
        json.dump(data, f, indent=2)
    
    with open(os.path.join(output_dir, 'correct_answers.txt'), 'w') as f:
        for label in labels:
            f.write(f"{label}\n")

def main():
    output_dir = 'output_directory'
    os.makedirs(output_dir, exist_ok=True)
    
    data_with_indices = load_social_iqa_data('dev.jsonl', num_samples=100)

    # store indices and items separately
    # prompt, type of question
    
    data_list = data_with_indices
    
    # Separate the data and labels into separate lists
    data_items = [item[0] for item in data_list]
    labels = [item[1] for item in data_list]

    # Print to verify
    # print("Data Items:", data_items)
    # print("Labels:", labels)

    generate_visual_social_iqa(data_items, output_dir)
    save_subset_info(data_items, labels, output_dir)

if __name__ == "__main__":
    main()


# import os
# import json
# import random
# from tqdm import tqdm
# from playwright.sync_api import sync_playwright

# def load_social_iqa_data(file_path, num_samples=100):
#     with open(file_path, 'r') as f:
#         data = [json.loads(line) for line in f]
#     return random.sample(data, num_samples)

# def load_labels(file_path):
#     with open(file_path, 'r') as f:
#         return [int(line.strip()) for line in f]

# def generate_html(item, variation):
#     template = """
#     <!DOCTYPE html>
#     <html lang="en">
#     <head>
#         <meta charset="UTF-8">
#         <meta name="viewport" content="width=device-width, initial-scale=1.0">
#         <title>Social IQA Question</title>
#         <style>
#             body {{
#                 font-family: Arial, sans-serif;
#                 background-color: #2c3e50;
#                 color: white;
#                 display: flex;
#                 justify-content: center;
#                 align-items: center;
#                 height: 100vh;
#                 margin: 0;
#                 padding: 20px;
#                 box-sizing: border-box;
#             }}
#             .container {{
#                 width: 100%;
#                 max-width: 600px;
#             }}
#             h2 {{
#                 margin-bottom: 20px;
#             }}
#             .option {{
#                 display: flex;
#                 align-items: center;
#                 margin: 10px 0;
#                 padding: 5px;
#                 border-radius: 5px;
#             }}
#             .radio {{
#                 width: 20px;
#                 height: 20px;
#                 border: 2px solid white;
#                 border-radius: 50%;
#                 margin-right: 10px;
#                 flex-shrink: 0;
#             }}
#             .text {{
#                 flex: 1;
#             }}
#             .selected {{
#                 background-color: #f1c40f;
#                 color: #2c3e50;
#             }}
#             .selected .radio {{
#                 background-color: #3498db;
#                 border-color: #3498db;
#             }}
#         </style>
#     </head>
#     <body>
#         <div class="container">
#             <h2>{context}</h2>
#             <p><strong>{question}</strong></p>
#             <div class="option{selected_a}">
#                 <div class="radio"></div>
#                 <div class="text">{answer_a}</div>
#             </div>
#             <div class="option{selected_b}">
#                 <div class="radio"></div>
#                 <div class="text">{answer_b}</div>
#             </div>
#             <div class="option{selected_c}">
#                 <div class="radio"></div>
#                 <div class="text">{answer_c}</div>
#             </div>
#         </div>
#     </body>
#     </html>
#     """
    
#     selected = {
#         'neutral': ['', '', ''],
#         'optionA': [' selected', '', ''],
#         'optionB': ['', ' selected', ''],
#         'optionC': ['', '', ' selected']
#     }
    
#     return template.format(
#         context=item['context'],
#         question=item['question'],
#         answer_a=item['answerA'],
#         answer_b=item['answerB'],
#         answer_c=item['answerC'],
#         selected_a=selected[variation][0],
#         selected_b=selected[variation][1],
#         selected_c=selected[variation][2]
#     )

# def html_to_png(html_content, output_path, width=800, height=600):
#     with sync_playwright() as p:
#         browser = p.chromium.launch()
#         page = browser.new_page()
#         page.set_content(html_content)
#         page.set_viewport_size({"width": width, "height": height})
#         page.screenshot(path=output_path)
#         browser.close()

# def generate_visual_social_iqa(data, output_dir):
#     variations = ['neutral', 'optionA', 'optionB', 'optionC']
    
#     for variation in variations:
#         os.makedirs(os.path.join(output_dir, f'v_social_i_qa_highlight_{variation}'), exist_ok=True)
    
#     for idx, item in enumerate(tqdm(data)):
#         for variation in variations:
#             html_content = generate_html(item, variation)
#             png_path = os.path.join(output_dir, f'v_social_i_qa_highlight_{variation}', f"question_{idx}.png")
#             html_to_png(html_content, png_path)

# def save_subset_info(data, labels, output_dir):
#     with open(os.path.join(output_dir, 'selected_subset.json'), 'w') as f:
#         json.dump(data, f, indent=2)
    
#     with open(os.path.join(output_dir, 'correct_answers.txt'), 'w') as f:
#         for label in labels:
#             f.write(f"{label}\n")

# def main():
#     output_dir = 'output_directory'
#     os.makedirs(output_dir, exist_ok=True)
    
#     data = load_social_iqa_data('dev.jsonl', num_samples=100)
#     labels = load_labels('dev-labels.lst')
#     labels = labels[:len(data)]  # Ensure we have the same number of labels as data items
    
#     generate_visual_social_iqa(data, output_dir)
#     save_subset_info(data, labels, output_dir)

# if __name__ == "__main__":
#     main()