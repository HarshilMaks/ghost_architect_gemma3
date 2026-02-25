"""
Ghost Architect: Synthetic Dataset Generator.
Uses a Teacher Model (Gemini Vision) to generate SQL schemas from UI screenshots.
"""
import os
import json
import time
from pathlib import Path
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- Configuration ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found in environment variables.\n"
        "Please create a .env file in the project root with:\n"
        "  GEMINI_API_KEY=your_actual_api_key_here\n"
        "See .env.example for template."
    )
genai.configure(api_key=GEMINI_API_KEY)

# Use Gemini 1.5 Flash (Fast, free, and excellent at vision)
model = genai.GenerativeModel('gemini-1.5-flash')

SCREENSHOT_DIR = "data/ui_screenshots"
DATASET_FILE = "data/dataset.json"

# The Prompt that turns the AI into a Senior Architect
SYSTEM_PROMPT = """
You are a Senior PostgreSQL Database Architect. 
Analyze the provided user interface screenshot and reverse-engineer the underlying database schema required to support it.

Follow these strict rules:
1. Identify all visible data entities (e.g., Users, Products, Orders, Analytics).
2. Infer the necessary fields, data types, and primary/foreign key relationships (1:1, 1:N, M:N).
3. Output ONLY valid PostgreSQL `CREATE TABLE` statements.
4. Do not include any explanations, greetings, or pleasantries. Output strictly the SQL code.
"""

def generate_sql_for_image(image_path):
    """Sends the image to the Teacher Model to get the SQL."""
    try:
        img = Image.open(image_path)
        # Send the prompt + image to Gemini
        response = model.generate_content([SYSTEM_PROMPT, img])
        
        # Clean up the response (remove markdown code blocks if the AI adds them)
        sql_output = response.text.replace("```sql", "").replace("```postgresql", "").replace("```", "").strip()
        return sql_output
    except Exception as e:
        print(f"  -> Error generating SQL for {image_path}: {e}")
        return None

def build_dataset():
    """Iterates through screenshots and builds the dataset.json file."""
    if not os.path.exists(SCREENSHOT_DIR):
        print(f"Error: {SCREENSHOT_DIR} not found. Run the download script first!")
        return

    # Load existing dataset so we can resume if the script stops
    dataset = []
    if os.path.exists(DATASET_FILE):
        with open(DATASET_FILE, "r") as f:
            dataset = json.load(f)
    
    # Keep track of images we already processed
    processed_images = {item.get("image_path") for item in dataset}
    
    # Get all PNG files
    image_files = list(Path(SCREENSHOT_DIR).glob("*.png"))
    print(f"Found {len(image_files)} screenshots. Starting extraction...")

    for img_path in image_files:
        img_str_path = str(img_path)
        
        if img_str_path in processed_images:
            continue
            
        print(f"\nAnalyzing: {img_path.name}")
        sql = generate_sql_for_image(img_str_path)
        
        if sql:
            # Create the exact training format required for Gemma-3
            training_example = {
                "instruction": "Analyze this UI screenshot and generate the PostgreSQL database schema required to support it.",
                "input": "", # For multimodal, the input is technically the image
                "output": sql,
                "image_path": img_str_path # We keep this reference for the multimodal dataloader later
            }
            
            dataset.append(training_example)
            
            # Save progressively so you don't lose data if it crashes
            with open(DATASET_FILE, "w") as f:
                json.dump(dataset, f, indent=2)
                
            print(f"  -> Success! Schema saved.")
            
            # Sleep to avoid hitting free-tier API rate limits (15 requests/minute for free tier)
            time.sleep(4) 

if __name__ == "__main__":
    build_dataset()