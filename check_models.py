import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("❌ No API Key found in .env")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models?key={API_KEY}"

try:
    response = requests.get(url, timeout=10)
    if response.status_code == 200:
        models = response.json().get('models', [])
        print(f"✅ Found {len(models)} models available for your key:")
        for m in models:
            if 'generateContent' in m.get('supportedGenerationMethods', []):
                print(f"   - {m['name']}")
    else:
        print(f"❌ Failed to list models. Status: {response.status_code}")
        print(response.text)
except Exception as e:
    print(f"⚠️ Exception: {e}")
