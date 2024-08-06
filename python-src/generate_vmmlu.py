import os
import json
from PIL import Image, ImageDraw, ImageFont
import textwrap
from datasets import load_dataset

def generate_vmmlu_variations(output_dir):
    tiny_data_test = load_dataset('tinyBenchmarks/tinyMMLU')['test']
    
    variations = ['neutral', 'optionA', 'optionB', 'optionC', 'optionD']
    
    for variation in variations:
        os.makedirs(os.path.join(output_dir, f'vmmlu_{variation}'), exist_ok=True)
    
    for idx, item in enumerate(tiny_data_test):
        for variation in variations:
            #print("item", item)
            html_content = generate_html(item, variation)
            image = generate_image(item, variation)
            
            # Save HTML
            with open(os.path.join(output_dir, f'vmmlu_{variation}', f"question_{idx}.html"), 'w') as f:
                f.write(html_content)
            
            # Save image
            image.save(os.path.join(output_dir, f'vmmlu_{variation}', f"question_{idx}.png"))

def generate_html(item, variation):
    # Load the HTML template
    with open('question_template.html', 'r') as f:
        template = f.read()
    
    # Replace placeholders with actual content
    try:
        html_content = template.format(
            question=item['question'],
            choiceA=item['choices'][0],
            choiceB=item['choices'][1],
            choiceC=item['choices'][2],
            choiceD=item['choices'][3],
            checkedA='checked' if variation == 'optionA' else '',
            checkedB='checked' if variation == 'optionB' else '',
            checkedC='checked' if variation == 'optionC' else '',
            checkedD='checked' if variation == 'optionD' else ''
        )
    except KeyError as e:
        print(f"KeyError: {e}")
        print("Template content:", template)
        print("Data provided:", item)
        raise

    return html_content

def draw_choices(d, item, variation, font, y_text):
    for i, choice in enumerate(item['choices']):
        is_selected = variation != 'neutral' and chr(65 + i) == variation[-1]
        
        # Draw the filled bubble
        bubble_x = 20
        bubble_y = y_text + 2
        bubble_size = 25
        d.ellipse([bubble_x, bubble_y, bubble_x + bubble_size, bubble_y + bubble_size], 
                  fill='white' if is_selected else '#2c3e50', 
                  outline='white', width=2)
        
        # Draw the answer symbol (e.g., "A", "B", etc.)
        # symbol = chr(65 + i)
        # symbol_bbox = d.textbbox((0, 0), symbol, font=font)
        # symbol_width = (symbol_bbox[2] - symbol_bbox[0]) - 10
        # symbol_height = (symbol_bbox[3] - symbol_bbox[1]) - 10
        # print("symbol_width", symbol_width)
        # print("symbol_height", symbol_height)
        # symbol_x = bubble_x + (bubble_size - symbol_width) // 2
        # symbol_y = bubble_y + (bubble_size - symbol_height) // 2
        # d.text((symbol_x, symbol_y), symbol, fill='#2c3e50' if is_selected else 'white', font=font)
        
        # Wrap and draw the choice text
        wrapped_choice = textwrap.wrap(choice, width=65)
        for j, line in enumerate(wrapped_choice):
            line_bbox = d.textbbox((0, 0), line, font=font)
            line_height = line_bbox[3] - line_bbox[1]
            
            # Calculate text position
            text_x = 55  # Adjust this value to position text after the bubble
            text_y = y_text + j * (line_height + 5)
            
            # Draw the text
            d.text((text_x, text_y), line, fill='white', font=font)
        
        # Update y_text for the next choice
        choice_height = max(bubble_size, (len(wrapped_choice) * (line_height + 5)) - 5)
        y_text += choice_height + 15  # Add space between choices
    
    return y_text

def generate_image(item, variation):
    # Create a blank image with higher resolution
    img = Image.new('RGB', (1280, 1400), color='#2c3e50')
    d = ImageDraw.Draw(img)
    
    # Try to use a system font with larger size
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 36)
        small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 32)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 36)
            small_font = ImageFont.truetype("arial.ttf", 32)
        except OSError:
            font = ImageFont.load_default()
            small_font = font
    
    # Wrap the question text
    wrapped_question = textwrap.wrap(item['question'], width=70)
    
    # Draw the question
    y_text = 40
    for line in wrapped_question:
        bbox = d.textbbox((0, 0), line, font=font)
        line_height = bbox[3] - bbox[1]
        d.text((40, y_text), line, fill='white', font=font)
        y_text += line_height + 10  # Add some padding between lines
    
    y_text += 40  # Add some padding before answers

    # Draw choices
    y_text = draw_choices(d, item, variation, small_font, y_text)
    
    # Resize the image to 560x560 with antialiasing
    img = img.resize((560, 640), Image.LANCZOS)
    
    return img

# Usage
generate_vmmlu_variations('output_directory')
