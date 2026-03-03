"""
Data Factory: Generate 5,000+ synthetic UI/SQL pairs using Gemini + Playwright.

Pipeline:
  1. Generate UI descriptions (JSON) using Gemini Free API
  2. Render descriptions to HTML using Tailwind CSS
  3. Screenshot HTML with Playwright
  4. Generate SQL schemas from screenshots using Gemini Vision
"""

import os
import json
import asyncio
import random
from pathlib import Path
from typing import List, Dict, Any
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY not found in environment variables")

genai.configure(api_key=GEMINI_API_KEY)

# Directories
OUTPUT_DIR = Path("data/synthetic_factory")
UI_DESCRIPTIONS_FILE = OUTPUT_DIR / "ui_descriptions.json"
HTML_DIR = OUTPUT_DIR / "html_pages"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
DATASET_FILE = OUTPUT_DIR / "synthetic_dataset.json"

# Create directories
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

# Get Gemini model
def get_gemini_model():
    """Get the best available Gemini model."""
    try:
        return genai.GenerativeModel('gemini-2.5-flash')
    except:
        try:
            return genai.GenerativeModel('gemini-2.0-flash')
        except:
            return genai.GenerativeModel('gemini-1.5-flash')


# ============================================================================
# STEP 1: Generate UI Descriptions
# ============================================================================

def generate_ui_descriptions(num_variations: int = 100) -> List[Dict[str, Any]]:
    """Generate diverse UI descriptions using Gemini."""
    print(f"\n🎨 STEP 1: Generating {num_variations} UI descriptions...")
    
    # Check if already generated
    if UI_DESCRIPTIONS_FILE.exists():
        with open(UI_DESCRIPTIONS_FILE, 'r') as f:
            existing = json.load(f)
            if len(existing) >= num_variations:
                print(f"  ✅ Already have {len(existing)} descriptions. Skipping generation.")
                return existing
    
    model = get_gemini_model()
    descriptions = []
    
    # Diverse UI types to generate
    ui_types = [
        "User Profile Page with avatar, bio, and stats",
        "E-commerce Product Listing with filters",
        "Admin Dashboard with charts and metrics",
        "Todo/Task Management application",
        "Social Media Feed with posts and comments",
        "Customer Support Ticket System",
        "Analytics Dashboard with time-series data",
        "User Settings/Preferences Panel",
        "Search Results Page",
        "Calendar/Scheduling Interface",
        "Invoice/Receipt Layout",
        "Contact/Address Book",
        "File Manager/Explorer",
        "Music/Media Player Library",
        "Restaurant Menu/Ordering System",
    ]
    
    batches = max(1, num_variations // len(ui_types))
    
    for ui_type in ui_types:
        for batch in range(batches):
            prompt = f"""
Generate a JSON description of a user interface for: "{ui_type}" (Variation {batch + 1})

Requirements:
1. Create a realistic, detailed description of what this UI would contain
2. Include specific field names, colors, layout structure
3. Return ONLY valid JSON (no markdown, no explanations)
4. Include these fields in JSON:
   - name: short name
   - type: category (e.g., "dashboard", "form", "listing")
   - description: detailed text description of the UI
   - fields: list of form fields or data columns (if applicable)
   - layout: "single-column", "two-column", "three-column", "card-grid"
   - has_charts: true/false
   - has_forms: true/false
   - color_scheme: "light", "dark", or specific colors

Make each variation unique with different fields, layouts, and features.
"""
            
            try:
                response = model.generate_content(prompt)
                
                # Try to parse JSON
                json_str = response.text
                if "```json" in json_str:
                    json_str = json_str.split("```json")[1].split("```")[0]
                elif "```" in json_str:
                    json_str = json_str.split("```")[1].split("```")[0]
                
                ui_desc = json.loads(json_str.strip())
                descriptions.append(ui_desc)
                print(f"  ✅ Generated: {ui_desc.get('name', 'Unknown')}")
                
                # Respect rate limits
                time.sleep(1)
                
            except Exception as e:
                print(f"  ⚠️  Failed to generate {ui_type} variation {batch}: {str(e)[:100]}")
                continue
    
    # Save descriptions
    with open(UI_DESCRIPTIONS_FILE, 'w') as f:
        json.dump(descriptions, f, indent=2)
    
    print(f"\n  📊 Saved {len(descriptions)} UI descriptions to {UI_DESCRIPTIONS_FILE}")
    return descriptions


# ============================================================================
# STEP 2: Convert descriptions to HTML
# ============================================================================

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ ui.name }}</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu; }
    </style>
</head>
<body class="bg-gray-50">
    <div class="min-h-screen">
        <!-- Header -->
        <header class="bg-white border-b border-gray-200 px-6 py-4 shadow-sm">
            <h1 class="text-2xl font-bold text-gray-900">{{ ui.name }}</h1>
            <p class="text-sm text-gray-600 mt-1">{{ ui.description[:100] }}...</p>
        </header>
        
        <!-- Main Content -->
        <main class="max-w-7xl mx-auto px-6 py-8">
            {% if ui.layout == 'card-grid' %}
                <div class="grid grid-cols-3 gap-6">
                    {% for item in ui.get('fields', [])[:9] %}
                    <div class="bg-white rounded-lg border border-gray-200 p-4 shadow-sm hover:shadow-md transition">
                        <h3 class="font-semibold text-gray-900">{{ item }}</h3>
                        <p class="text-sm text-gray-500 mt-2">Sample data for {{ item }}</p>
                        <div class="mt-3 h-8 bg-gradient-to-r from-blue-100 to-blue-50 rounded"></div>
                    </div>
                    {% endfor %}
                </div>
            {% elif ui.layout == 'two-column' %}
                <div class="grid grid-cols-2 gap-6">
                    <div class="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
                        <h2 class="text-lg font-semibold text-gray-900 mb-4">Sidebar / Filters</h2>
                        {% for field in ui.get('fields', [])[:5] %}
                            <div class="mb-3 pb-3 border-b border-gray-100 last:border-0">
                                <label class="block text-sm font-medium text-gray-700">{{ field }}</label>
                                <input type="text" class="mt-1 w-full px-3 py-2 border border-gray-300 rounded-md text-sm" placeholder="Enter {{ field|lower }}">
                            </div>
                        {% endfor %}
                    </div>
                    <div class="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
                        <h2 class="text-lg font-semibold text-gray-900 mb-4">Main Content</h2>
                        {% for i in range(5) %}
                            <div class="mb-4 pb-4 border-b border-gray-100 last:border-0">
                                <div class="h-4 bg-gray-200 rounded w-3/4 mb-2"></div>
                                <div class="h-4 bg-gray-100 rounded w-1/2"></div>
                            </div>
                        {% endfor %}
                    </div>
                </div>
            {% elif ui.layout == 'three-column' %}
                <div class="grid grid-cols-3 gap-6">
                    {% for col in ['Left', 'Center', 'Right'] %}
                    <div class="bg-white rounded-lg border border-gray-200 p-4 shadow-sm">
                        <h3 class="font-semibold text-gray-900 mb-4">{{ col }} Panel</h3>
                        {% for i in range(3) %}
                            <div class="mb-3 h-12 bg-gradient-to-r from-gray-100 to-gray-50 rounded flex items-center px-3 text-sm text-gray-600">
                                Item {{ i + 1 }}
                            </div>
                        {% endfor %}
                    </div>
                    {% endfor %}
                </div>
            {% else %}
                <div class="bg-white rounded-lg border border-gray-200 p-6 shadow-sm">
                    <h2 class="text-lg font-semibold text-gray-900 mb-4">{{ ui.name }}</h2>
                    {% if ui.get('has_charts') %}
                        <div class="mb-6 h-64 bg-gradient-to-br from-blue-50 to-indigo-50 rounded-lg border border-blue-100 flex items-center justify-center">
                            <span class="text-gray-500">Chart Placeholder</span>
                        </div>
                    {% endif %}
                    {% if ui.get('has_forms') %}
                        <form class="space-y-4">
                            {% for field in ui.get('fields', [])[:8] %}
                                <div>
                                    <label class="block text-sm font-medium text-gray-700">{{ field }}</label>
                                    <input type="text" class="mt-1 w-full px-3 py-2 border border-gray-300 rounded-md" placeholder="Enter {{ field|lower }}">
                                </div>
                            {% endfor %}
                            <button type="submit" class="w-full bg-blue-600 hover:bg-blue-700 text-white font-medium py-2 rounded-md transition">Submit</button>
                        </form>
                    {% else %}
                        <div class="space-y-3">
                            {% for field in ui.get('fields', [])[:10] %}
                                <div class="flex items-center justify-between py-3 border-b border-gray-100">
                                    <span class="text-gray-700 font-medium">{{ field }}</span>
                                    <span class="text-gray-500 text-sm">Sample value</span>
                                </div>
                            {% endfor %}
                        </div>
                    {% endif %}
                </div>
            {% endif %}
        </main>
    </div>
</body>
</html>
"""

def generate_html_pages(descriptions: List[Dict]) -> List[str]:
    """Convert UI descriptions to HTML files."""
    print(f"\n🌐 STEP 2: Converting {len(descriptions)} descriptions to HTML...")
    
    try:
        from jinja2 import Template
    except ImportError:
        print("  ⚠️  jinja2 not installed. Installing...")
        os.system("pip install jinja2")
        from jinja2 import Template
    
    template = Template(HTML_TEMPLATE)
    html_files = []
    
    for i, ui_desc in enumerate(descriptions):
        try:
            html_content = template.render(ui=ui_desc)
            html_file = HTML_DIR / f"ui_{i:04d}.html"
            
            with open(html_file, 'w') as f:
                f.write(html_content)
            
            html_files.append(str(html_file))
            
            if (i + 1) % 20 == 0:
                print(f"  ✅ Generated {i + 1} HTML pages")
                
        except Exception as e:
            print(f"  ❌ Failed to generate HTML for {ui_desc.get('name')}: {str(e)[:100]}")
    
    print(f"  📊 Total HTML pages generated: {len(html_files)}")
    return html_files


# ============================================================================
# STEP 3: Screenshot HTML with Playwright
# ============================================================================

async def screenshot_html_pages(html_files: List[str]) -> List[str]:
    """Screenshot HTML pages using Playwright."""
    print(f"\n📸 STEP 3: Screenshotting {len(html_files)} HTML pages...")
    
    try:
        from playwright.async_api import async_playwright
    except ImportError:
        print("  ⚠️  playwright not installed. Installing...")
        os.system("pip install playwright")
        os.system("playwright install")
        from playwright.async_api import async_playwright
    
    screenshots = []
    
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 720})
        
        for i, html_file in enumerate(html_files):
            try:
                # Convert to file:// URL
                file_url = Path(html_file).absolute().as_uri()
                await page.goto(file_url, wait_until="networkidle")
                
                # Screenshot
                screenshot_file = SCREENSHOTS_DIR / f"ui_{i:04d}.png"
                await page.screenshot(path=str(screenshot_file), full_page=False)
                screenshots.append(str(screenshot_file))
                
                if (i + 1) % 50 == 0:
                    print(f"  ✅ Screenshotted {i + 1} pages")
                    
            except Exception as e:
                print(f"  ⚠️  Failed to screenshot {html_file}: {str(e)[:100]}")
        
        await browser.close()
    
    print(f"  📊 Total screenshots generated: {len(screenshots)}")
    return screenshots


# ============================================================================
# STEP 4: Generate SQL schemas (using existing synthetic_generator logic)
# ============================================================================

def generate_sql_from_screenshots(screenshot_files: List[str]) -> List[Dict]:
    """Generate SQL schemas from screenshots."""
    print(f"\n🗄️  STEP 4: Generating SQL schemas from {len(screenshot_files)} screenshots...")
    
    model = get_gemini_model()
    
    SYSTEM_PROMPT = """
You are a Senior PostgreSQL Database Architect.
Analyze the provided UI screenshot and generate the optimal database schema.

Rules:
1. Identify all visible data entities
2. Infer fields, data types, and relationships
3. Output ONLY valid PostgreSQL CREATE TABLE statements
4. No explanations, just SQL code
"""
    
    dataset = []
    
    for i, screenshot_file in enumerate(screenshot_files):
        try:
            img = Image.open(screenshot_file)
            response = model.generate_content([SYSTEM_PROMPT, img])
            
            sql = response.text.replace("```sql", "").replace("```postgresql", "").replace("```", "").strip()
            
            dataset.append({
                "image_path": screenshot_file,
                "instruction": "Analyze this UI and generate the PostgreSQL schema.",
                "output": sql,
                "domain": "synthetic",
                "size_kb": round(Path(screenshot_file).stat().st_size / 1024, 1)
            })
            
            if (i + 1) % 50 == 0:
                print(f"  ✅ Generated SQL for {i + 1} screenshots")
            
            # Respect rate limits
            time.sleep(1)
            
        except Exception as e:
            print(f"  ⚠️  Failed to generate SQL for {screenshot_file}: {str(e)[:100]}")
    
    # Save dataset
    with open(DATASET_FILE, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"  📊 Saved {len(dataset)} SQL schemas to {DATASET_FILE}")
    return dataset


# ============================================================================
# Main Pipeline
# ============================================================================

async def run_data_factory(num_ui_descriptions: int = 100):
    """Run the complete data factory pipeline."""
    print("\n" + "="*80)
    print("🏭 GHOST ARCHITECT DATA FACTORY")
    print("="*80)
    
    # Step 1: Generate UI descriptions
    descriptions = generate_ui_descriptions(num_ui_descriptions)
    
    if not descriptions:
        print("❌ No descriptions generated. Exiting.")
        return
    
    # Step 2: Generate HTML
    html_files = generate_html_pages(descriptions)
    
    if not html_files:
        print("❌ No HTML generated. Exiting.")
        return
    
    # Step 3: Screenshot
    screenshots = await screenshot_html_pages(html_files)
    
    if not screenshots:
        print("❌ No screenshots generated. Exiting.")
        return
    
    # Step 4: Generate SQL
    dataset = generate_sql_from_screenshots(screenshots)
    
    print("\n" + "="*80)
    print("✅ DATA FACTORY COMPLETE!")
    print(f"   Generated {len(dataset)} synthetic UI/SQL pairs")
    print(f"   📁 Saved to: {OUTPUT_DIR}")
    print(f"   📊 Dataset: {DATASET_FILE}")
    print("="*80 + "\n")
    
    return dataset


if __name__ == "__main__":
    import sys
    
    num_variations = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    
    # Run async function
    asyncio.run(run_data_factory(num_variations))
