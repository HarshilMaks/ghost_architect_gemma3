# Security Guide: API Key Management

## ⚠️ Critical: Never Commit API Keys to GitHub

Your project contains scripts that use external APIs (Google Gemini). **Never commit your actual API keys.**

---

## Setup Instructions

### 1. Copy the Template
```bash
cp .env.example .env
```

### 2. Add Your API Keys
Edit `.env` and fill in your actual keys:
```
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 3. Get API Keys

#### Google Gemini Vision API
1. Go to https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy the key and paste into `.env`
4. Free tier: 60 requests/minute, unlimited requests/day

#### (Optional) OpenAI API
1. Go to https://platform.openai.com/api-keys
2. Create new API key
3. Add to `.env` as `OPENAI_API_KEY`

### 4. Verify Setup
```bash
# Test that your script can load the API key
uv run python src/synthetic_generator.py
```

If successful, you'll see: `Found X screenshots. Starting extraction...`

---

## Security Best Practices

✅ **Do**
- Use `.env` for all sensitive credentials
- Keep `.env.example` in repo (shows template only)
- Use `load_dotenv()` to load from `.env`
- Rotate API keys regularly
- Monitor API usage in your cloud console

❌ **Don't**
- Hardcode API keys in Python files
- Commit `.env` to Git (it's in `.gitignore`)
- Share your API keys in GitHub Issues/Discussions
- Use the same key across multiple projects
- Log API keys in error messages

---

## File Layout

```
ghost_architect_gemma3/
├── .env                 # ⚠️  YOUR ACTUAL KEYS (NEVER COMMIT)
├── .env.example         # ✅ Template (safe to commit)
├── .gitignore           # Already includes .env
└── src/
    └── synthetic_generator.py  # Loads from .env
```

---

## Troubleshooting

### Error: "GEMINI_API_KEY not found in environment variables"
**Solution:** Create `.env` file (copy from `.env.example`) and add your key.

### Error: "Invalid API key"
**Solution:** 
- Copy your key carefully (no extra spaces)
- Verify key hasn't expired in Google AI Studio
- Generate a new key and try again

### Error: "Quota exceeded"
**Solution:**
- Free tier: 60 requests/minute
- Wait 1 minute before retrying
- Upgrade to paid plan for higher limits

### I accidentally committed my .env file!
**Solution** (⚠️ Immediate action required):
1. Rotate all API keys immediately
2. Remove the file from Git history:
   ```bash
   git rm --cached .env
   git commit -m "Remove accidentally committed .env file"
   git push
   ```
3. Generate new API keys and update `.env`

---

## Using the API in Code

**❌ Bad (NEVER do this):**
```python
API_KEY = "sk-abc123..."  # Hardcoded!
```

**✅ Good (do this):**
```python
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise ValueError("API_KEY not found in .env")
```

All scripts in this project follow the **✅ Good** pattern.

---

## GitHub Security Scanning

GitHub automatically scans for exposed credentials:
- If you accidentally push an API key, GitHub will notify you
- Rotate the key immediately
- Remove from Git history using `git filter-branch` or `bfg`

---

## Questions?

See:
- `.env.example` — Template for all required keys
- `src/synthetic_generator.py` — Example of secure API loading
- `requirements.txt` — python-dotenv package details
