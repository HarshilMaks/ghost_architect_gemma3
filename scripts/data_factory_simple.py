"""
Simplified Data Factory: Skip screenshots, use cached or placeholder images
"""
import os
import json
from pathlib import Path
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Set GEMINI_API_KEY in .env")

genai.configure(api_key=GEMINI_API_KEY)

OUTPUT_DIR = Path("data/synthetic_factory")
DATASET_FILE = OUTPUT_DIR / "synthetic_dataset.json"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def generate_ui_descriptions(count: int = 100) -> List[Dict]:
    """Generate diverse UI descriptions."""
    print(f"🎨 Generating {count} UI descriptions...")
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    ui_types = [
        "Admin Dashboard with analytics and KPIs",
        "E-commerce Product Catalog with filters",
        "User Profile Management System",
        "Social Media Feed with comments",
        "Data Table with sorting and pagination",
        "Chat/Messaging Interface",
        "Calendar Event Management",
        "Billing & Invoice System",
        "Settings & Preferences Panel",
        "Search Results Page"
    ]
    
    descriptions = []
    for i in range(count):
        ui_type = ui_types[i % len(ui_types)]
        prompt = f"""Generate a detailed UI/database schema specification for: {ui_type}
        
Respond ONLY with valid JSON (no markdown, no code blocks):
{{
  "title": "UI Name",
  "description": "What this UI does",
  "components": ["list", "of", "UI", "components"],
  "key_tables": ["table1", "table2"]
}}"""
        
        try:
            response = model.generate_content(prompt)
            try:
                data = json.loads(response.text)
                data['id'] = i
                descriptions.append(data)
                if (i + 1) % 10 == 0:
                    print(f"   ✅ {i + 1}/{count}")
            except:
                descriptions.append({
                    "id": i,
                    "title": ui_type,
                    "description": "Auto-generated UI specification",
                    "components": ["header", "content", "footer"],
                    "key_tables": ["users", "products"]
                })
        except Exception as e:
            print(f"   ⚠️  Skipped {i}: {str(e)[:50]}")
    
    return descriptions

def generate_sql_schemas(count: int, descriptions: List[Dict]) -> List[Dict]:
    """Generate SQL schemas from UI descriptions."""
    print(f"\n📊 Generating {count} SQL schemas...")
    model = genai.GenerativeModel('gemini-2.0-flash')
    
    dataset = []
    for i, desc in enumerate(descriptions[:count]):
        prompt = f"""Generate a CREATE TABLE SQL schema for this UI: {desc.get('title')}
        
Requirements:
- Use snake_case column names
- Include id, created_at, updated_at columns
- Return ONLY the SQL, no explanation

UI Tables: {', '.join(desc.get('key_tables', ['users', 'data']))}"""
        
        try:
            response = model.generate_content(prompt)
            sql = response.text.strip()
            
            dataset.append({
                "image_path": f"synthetic_{i:04d}.png",
                "domain": "database_schema",
                "instruction": f"Generate SQL for: {desc.get('title')}",
                "output": sql,
                "ui_type": desc.get('title', 'unknown')
            })
            
            if (i + 1) % 10 == 0:
                print(f"   ✅ {i + 1}/{len(descriptions)}")
        except Exception as e:
            print(f"   ⚠️  Skipped {i}: {str(e)[:50]}")
    
    return dataset

def main(count: int = 100):
    """Generate synthetic dataset without screenshots."""
    print("\n" + "="*60)
    print(f"🏭 DATA FACTORY (No Screenshots): Generating {count} UI/SQL pairs")
    print("="*60)
    
    descs = generate_ui_descriptions(count)
    if not descs:
        print("❌ Failed to generate descriptions")
        return
    
    dataset = generate_sql_schemas(len(descs), descs)
    
    DATASET_FILE.write_text(json.dumps(dataset, indent=2))
    
    print("\n" + "="*60)
    print(f"✅ COMPLETE: Generated {len(dataset)} synthetic UI/SQL pairs")
    print(f"   📁 {DATASET_FILE}")
    print("="*60 + "\n")

if __name__ == "__main__":
    import sys
    count = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    main(count)
