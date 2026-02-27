"""
Ghost Architect: Synthetic Dataset Generator.
Uses a Teacher Model (Gemini Vision) to generate SQL schemas from UI screenshots.
Includes Smart Resume to save API limits.
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

# Use Gemini 2.5 Flash (Latest, fastest, best for vision tasks)
try:
    model = genai.GenerativeModel('gemini-2.5-flash')
    print("✅ Using gemini-2.5-flash")
except Exception as e:
    print(f"⚠️  gemini-2.5-flash not available, trying gemini-2.0-flash: {e}")
    try:
        model = genai.GenerativeModel('gemini-2.0-flash')
        print("✅ Using gemini-2.0-flash")
    except Exception as e2:
        print(f"⚠️  gemini-2.0-flash not available, trying gemini-1.5-flash: {e2}")
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            print("✅ Using gemini-1.5-flash")
        except Exception as e3:
            print(f"⚠️  gemini-1.5-flash not available, trying gemini-1.5-pro: {e3}")
            model = genai.GenerativeModel('gemini-1.5-pro')

SCREENSHOT_DIR = "data/ui_screenshots"
DATASET_FILE = "data/dataset_vision.json"

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

def generate_sql_for_image(image_path, verbose=False):
    """Sends the image to the Teacher Model to get the SQL."""
    try:
        img = Image.open(image_path)
        # Send the prompt + image to Gemini
        response = model.generate_content([SYSTEM_PROMPT, img])
        
        # Clean up the response
        sql_output = response.text.replace("```sql", "").replace("```postgresql", "").replace("```", "").strip()
        return sql_output
    except Exception as e:
        error_msg = str(e)
        if verbose:
            print(f"  ❌ ERROR: {error_msg}")
        # Only log key errors, skip rate limit errors
        if "429" not in error_msg and "quota" not in error_msg.lower():
            print(f"  ⚠️  {os.path.basename(image_path)}: {error_msg[:80]}")
        return None

def build_dataset():
    """Iterates through screenshots and builds the dataset.json file."""
    if not os.path.exists(SCREENSHOT_DIR):
        print(f"Error: {SCREENSHOT_DIR} not found. Run the download script first!")
        return

    # Load existing dataset so we can resume if the script stops
    dataset = []
    if os.path.exists(DATASET_FILE):
        try:
            with open(DATASET_FILE, "r", encoding="utf-8") as f:
                dataset = json.load(f)
        except json.JSONDecodeError:
            print("⚠️  Warning: dataset.json was empty or corrupted. Starting fresh.")

    # SMART RESUME: Keep track of images we already processed (Handles both key formats)
    processed_images = set()
    for item in dataset:
        img_val = item.get("image_path") or item.get("image")
        if img_val:
            processed_images.add(img_val.replace("\\", "/"))
            
    print(f"🧠 Smart Resume: Skipping {len(processed_images)} images already generated...")
    
    # Get all PNG files
    image_files = list(Path(SCREENSHOT_DIR).glob("*.png"))
    print(f"📁 Found {len(image_files)} total screenshots in folder.")

    processed_this_session = 0
    
    for img_path in image_files:
        img_str_path = str(img_path).replace("\\", "/")
        
        if img_str_path in processed_images:
            continue
            
        print(f"\nAnalyzing: {img_path.name}")
        sql = generate_sql_for_image(img_str_path)
        
        if sql:
            # Check if this image already exists in dataset (upgrade its output)
            existing = next((ex for ex in dataset if (ex.get("image_path") or ex.get("image","")).replace("\\","/") == img_str_path), None)
            
            if existing:
                # Upgrade existing heuristic SQL with Gemini's better output
                existing["output"] = sql
                existing["instruction"] = "Analyze this UI screenshot and generate the PostgreSQL database schema required to support it."
            else:
                # New entry — add with full schema matching dataset_vision.json format
                dataset.append({
                    "image_path": img_str_path,
                    "instruction": "Analyze this UI screenshot and generate the PostgreSQL database schema required to support it.",
                    "output": sql,
                    "domain": "unknown",
                    "hash": "",
                    "size_kb": round(img_path.stat().st_size / 1024, 1)
                })
            
            # Save progressively so you don't lose data if it crashes
            with open(DATASET_FILE, "w", encoding="utf-8") as f:
                json.dump(dataset, f, indent=2)
                
            processed_this_session += 1
            print(f"  -> {'Upgraded' if existing else 'Added'}! Schema saved. ({processed_this_session} processed this run)")
            
            # Sleep to avoid hitting free-tier API rate limits (15 requests/minute)
            time.sleep(4) 

if __name__ == "__main__":
    import sys
    
    # Check if running in test mode
    test_mode = "--test" in sys.argv
    
    if test_mode:
        print("\n🧪 TEST MODE: Analyzing a single screenshot to verify setup")
        image_files = list(Path(SCREENSHOT_DIR).glob("*.png"))
        if image_files:
            test_img = image_files[0]
            print(f"Testing with: {test_img.name}")
            sql = generate_sql_for_image(str(test_img), verbose=True)
            if sql:
                print(f"✅ SUCCESS! Generated {len(sql)} chars of SQL")
                print(f"Preview: {sql[:200]}...")
            else:
                print("❌ FAILED: No SQL generated")
                print("\n💡 NOTE: Free tier Gemini API has quota limits (15 req/min, 1M tokens/day)")
        else:
            print(f"❌ No screenshots found in {SCREENSHOT_DIR}")
    else:
        print("\n📊 FULL MODE: Processing screenshots")
        print(f"Run with --test flag to verify single screenshot first:")
        print(f"  uv run python src/synthetic_generator.py --test")
        print("\n⚠️  WARNING: This requires many API calls. Free tier has quota limits.")
        build_dataset()