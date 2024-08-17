import os
from playwright.sync_api import sync_playwright
from tqdm import tqdm


def html_to_png(html_file_path, output_path, width=1280, height=720, quality=100):
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        # Load the HTML content
        with open(html_file_path, 'r', encoding='utf-8') as file:
            html_content = file.read()
        
        # Set the content of the page
        page.set_content(html_content)
        
        # Set viewport size
        page.set_viewport_size({"width": width, "height": height})
        
        # Wait for content to load
        page.wait_for_load_state('networkidle')
        
        # Take a screenshot
        page.screenshot(path=output_path, full_page=True)
        
        browser.close()

variations = ['neutral', 'optionA', 'optionB', 'optionC', 'optionD']

for variation in tqdm(variations):
    for i in tqdm(range(0, 100)):
        html_file_path = f'output_directory/vmmlu_{variation}_size/question_{i}.html'
        output_path = f'output_directory/vmmlu_{variation}_size_rendered/question_{i}.png'
        html_to_png(html_file_path, output_path)