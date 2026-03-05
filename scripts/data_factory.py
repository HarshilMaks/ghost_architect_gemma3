"""
Ghost Architect: Professional Batch Data Factory (v2.5) - STABLE RESUME
Generates 5,000 UI+SQL pairs by rotating through ALL verified Gemini models.
- Fixed 404 model names (using -latest aliases)
- Auto-skip on JSON errors
- Exponential backoff & Quota rotation
"""

import os
import json
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any
import google.generativeai as genai
from PIL import Image
from dotenv import load_dotenv
from jinja2 import Template

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY in .env")

genai.configure(api_key=GEMINI_API_KEY)

OUTPUT_DIR = Path("data/synthetic_factory")
HTML_DIR = OUTPUT_DIR / "html_pages"
SCREENSHOTS_DIR = OUTPUT_DIR / "screenshots"
DATASET_FILE = OUTPUT_DIR / "synthetic_dataset.json"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOTS_DIR.mkdir(parents=True, exist_ok=True)

HTML_TEMPLATE = """<!DOCTYPE html><html><head><script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-gray-50 p-8"><div class="max-w-4xl mx-auto bg-white shadow-xl rounded-xl overflow-hidden border">
<header class="bg-indigo-600 px-6 py-4 text-white"><h1 class="text-2xl font-bold">{{ name }}</h1></header>
<main class="p-6">{% if layout == 'card-grid' %}<div class="grid grid-cols-3 gap-4">
{% for f in fields %}<div class="p-4 border rounded-lg bg-gray-50"><p class="text-xs text-gray-400 uppercase font-bold">{{ f }}</p>
<div class="h-4 bg-gray-200 rounded mt-2 w-3/4"></div></div>{% endfor %}</div>
{% else %}<div class="space-y-4">{% for f in fields %}<div class="flex items-center justify-between p-3 border-b">
<span class="font-medium text-gray-700">{{ f }}</span><div class="h-6 w-24 bg-indigo-100 rounded"></div></div>{% endfor %}</div>{% endif %}
</main></div></body></html>"""

def get_available_models():
    """Return verified list of models based on user's check_models output."""
    return [
        'gemini-2.0-flash',
        'gemini-2.5-flash',
        'gemini-flash-lite-latest',
        'gemini-flash-latest',
        'gemini-pro-latest',
        'gemini-3.1-flash-lite-preview',
        'gemini-2.0-flash-lite-001',
        'gemini-2.5-pro'
    ]

def clean_json(text: str) -> str:
    """Robustly extract JSON from model response."""
    for delim in ['```json', '```']:
        if delim in text:
            text = text.split(delim)[1]
            if '```' in text:
                text = text.split('```')[0]
            break
    return text.strip()

async def main(target_count: int):
    print(f"\n🚀 Launching SOTA Data Factory (v2.5 Stable): Target {target_count} samples")
    
    models = get_available_models()
    current_model_idx = 0
    dataset = []
    
    backoff_time = 5 

    if DATASET_FILE.exists():
        try:
            with open(DATASET_FILE) as f:
                dataset = json.load(f)
            print(f"   Reusing {len(dataset)} existing samples")
        except:
            dataset = []

    ui_types = ["Dashboard", "Analytics", "CRM", "E-commerce", "SaaS", "Settings", "Profile", "Inventory"]
    template = Template(HTML_TEMPLATE)

    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1280, "height": 720})

        while len(dataset) < target_count:
            model_name = models[current_model_idx % len(models)]
            model = genai.GenerativeModel(model_name)
            ui_type = ui_types[len(dataset) % len(ui_types)]
            prompt = f"Generate 5 unique UI designs for {ui_type}. Return JSON list: [{{'name':'...', 'fields':[], 'layout':'card-grid|list', 'sql':'CREATE TABLE...'}}]"
            
            try:
                print(f"   🛰️  Requesting batch (Current: {len(dataset)}/{target_count}) using {model_name}...")
                resp = model.generate_content(prompt)
                batch_text = clean_json(resp.text)
                batch = json.loads(batch_text)

                for item in batch:
                    if len(dataset) >= target_count: break
                    idx = len(dataset)
                    file_name = f"ui_{idx:04d}"
                    
                    html = template.render(
                        name=item.get('name', 'App'), 
                        fields=item.get('fields', []), 
                        layout=item.get('layout', 'list')
                    )
                    html_path = HTML_DIR / f"{file_name}.html"
                    html_path.write_text(html)

                    ss_path = SCREENSHOTS_DIR / f"{file_name}.png"
                    await page.goto(html_path.absolute().as_uri())
                    await page.screenshot(path=str(ss_path))

                    dataset.append({
                        "image_path": str(ss_path),
                        "instruction": "Analyze the UI structure and generate an appropriate database schema.",
                        "output": item.get('sql', 'CREATE TABLE error (id INT);'),
                        "domain": ui_type,
                        "size_kb": round(ss_path.stat().st_size / 1024, 1)
                    })
                
                with open(DATASET_FILE, 'w') as f:
                    json.dump(dataset, f, indent=2)
                
                print(f"   ✅ Batch Complete. Total: {len(dataset)}")
                backoff_time = 5 
                await asyncio.sleep(2) 

            except Exception as e:
                error_msg = str(e)
                print(f"   ⚠️  Error with {model_name}: {error_msg[:80]}")
                
                if "429" in error_msg or "quota" in error_msg.lower():
                    print(f"   🔄 Quota hit! Switching model...")
                    current_model_idx += 1
                    if current_model_idx % len(models) == 0:
                        print(f"   ⏳ All models cycled. Daily limit likely reached. Sleeping 60s...")
                        await asyncio.sleep(60)
                else:
                    print(f"   ⏳ Retrying in {backoff_time}s...")
                    await asyncio.sleep(backoff_time)
                    backoff_time = min(backoff_time * 2, 60)

        await browser.close()

if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    asyncio.run(main(count))
