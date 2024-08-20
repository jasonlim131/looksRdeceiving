import os
import json
from PIL import Image, ImageDraw, ImageFont
import textwrap
from datasets import load_dataset

def generate_vmmlu_variations(output_dir):
    tiny_data_test = load_dataset('tinyBenchmarks/tinyMMLU')['test']
    
    variations = ['neutral', 'optionA', 'optionB', 'optionC', 'optionD']
    
    for variation in variations:
        os.makedirs(os.path.join(output_dir, f'vmmlu_{variation}_size'), exist_ok=True)
    
    for idx, item in enumerate(tiny_data_test):
        for variation in variations:
            html_content = generate_html(item, variation)
            image = generate_image(item, variation)
            
            # Save HTML
            with open(os.path.join(output_dir, f'vmmlu_{variation}_size', f"question_{idx}.html"), 'w') as f:
                f.write(html_content)
            
            # Save image
            image.save(os.path.join(output_dir, f'vmmlu_{variation}_size', f"question_{idx}.png"))
            
def generate_html(item, variation):
    # Load the HTML template
    with open('question_template_size.html', 'r') as f:
        template = f.read()
    
    # Define the CSS for different font sizes
    normal_css = """
        .choice-{0} {{
            margin: 10px 0;
            padding: 10px;
            background-color: #34495e;
            border-radius: 5px;
            font-size: 16px;
        }}
    """
    
    large_css = """
        .choice-{0} {{
            margin: 10px 0;
            padding: 10px;
            background-color: #34495e;
            border-radius: 5px;
            font-size: 32px;
        }}
    """
    
    # Prepare CSS for each choice
    choice_css = {
        'A': normal_css.format('A'),
        'B': normal_css.format('B'),
        'C': normal_css.format('C'),
        'D': normal_css.format('D')
    }
    
    # Set larger font size for the biased option
    if variation != 'neutral':
        biased_option = variation[-1]
        choice_css[biased_option] = large_css.format(biased_option)
    
    # Combine all CSS
    all_css = ''.join(choice_css.values())
    
    # Replace placeholders with actual content
    try:
        html_content = template.format(
            question=item['question'],
            choiceA=item['choices'][0],
            choiceB=item['choices'][1],
            choiceC=item['choices'][2],
            choiceD=item['choices'][3],
            css=all_css
        )
    except KeyError as e:
        print(f"KeyError: {e}")
        print("Template content:", template)
        print("Data provided:", item)
        raise

    return html_content

def draw_choices(d, item, variation, font, large_font, y_text):
    for i, choice in enumerate(item['choices']):
        is_biased = variation != 'neutral' and chr(65 + i) == variation[-1]
        
        # Draw the empty bubble (no filling)
        bubble_x = 20
        bubble_y = y_text + 2
        bubble_size = 25
        d.ellipse([bubble_x, bubble_y, bubble_x + bubble_size, bubble_y + bubble_size], 
                  fill=None, outline='white', width=2)
        
        # Wrap and draw the choice text
        current_font = large_font if is_biased else font
        wrapped_choice = textwrap.wrap(choice, width=55 if is_biased else 65)
        for j, line in enumerate(wrapped_choice):
            line_bbox = d.textbbox((0, 0), line, font=current_font)
            line_height = line_bbox[3] - line_bbox[1]
            
            # Calculate text position
            text_x = 55  # Adjust this value to position text after the bubble
            text_y = y_text + j * (line_height + 5)
            
            # Draw the text
            d.text((text_x, text_y), line, fill='white', font=current_font)
        
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
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 32)
        large_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 64)
    except OSError:
        try:
            font = ImageFont.truetype("arial.ttf", 32)
            large_font = ImageFont.truetype("arial.ttf", 64)
        except OSError:
            font = ImageFont.load_default()
            large_font = font
    
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
    y_text = draw_choices(d, item, variation, font, large_font, y_text)
    
    # Resize the image to 560x560 with antialiasing
    img = img.resize((560, 640), Image.LANCZOS)
    
    return img

# Usage
generate_vmmlu_variations('output_directory')