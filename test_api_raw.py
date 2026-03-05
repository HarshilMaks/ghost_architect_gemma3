import os
import requests
import json
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("❌ No API Key found in .env")
    exit(1)

url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={API_KEY}"

headers = {
    "Content-Type": "application/json"
}

data = {
    "contents": [{
        "parts": [{"text": "Hello, explain 429 errors briefly."}]
    }]
}

print(f"📡 Testing API Key: {API_KEY[:5]}...{API_KEY[-4:]}")
print(f"🌐 Endpoint: {url.split('?')[0]}")

try:
    response = requests.post(url, headers=headers, json=data, timeout=10)
    
    print(f"🔍 Status Code: {response.status_code}")
    
    if response.status_code == 200:
        print("✅ Success! Response:")
        print(response.json()['candidates'][0]['content']['parts'][0]['text'][:100] + "...")
    else:
        print("❌ Failed. Response:")
        print(response.text)
        
except Exception as e:
    print(f"⚠️ Exception: {e}")
