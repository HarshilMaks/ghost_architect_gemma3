"""
Bulletproof UI Screenshot Generator for Ghost Architect.
Uses Playwright with Network-Idle waiting and Auto-Trashing of blank screens.
"""
import os
import time
import requests
from urllib.parse import urlparse
from playwright.sync_api import sync_playwright
from tqdm import tqdm

# --- Configuration ---
OUTPUT_DIR = "data/ui_screenshots"
MAX_URLS = 1000  # Let's grab 1000 high-quality screens
MIN_FILE_SIZE_KB = 40  # Anything under 40KB is usually a blank/broken screen

def get_fresh_github_urls():
    """Fetches paginated, modern URLs from GitHub API (2024+)."""
    print(f"Fetching up to {MAX_URLS} fresh, modern URLs from GitHub API (2024+)...")
    fresh_urls = set()
    
    search_queries = [
        "topic:dashboard pushed:>2024-01-01",
        "topic:admin-panel pushed:>2024-01-01",
        "topic:saas pushed:>2024-01-01",
        "topic:erp pushed:>2024-01-01"
    ]
    
    headers = {"Accept": "application/vnd.github.v3+json", "User-Agent": "GhostArchitect"}
    
    for query in search_queries:
        print(f"\n  -> Searching GitHub for: {query}")
        for page in range(1, 10):
            try:
                url = f"https://api.github.com/search/repositories?q={query}&sort=updated&order=desc&per_page=100&page={page}"
                response = requests.get(url, headers=headers)
                
                if response.status_code == 200:
                    items = response.json().get("items", [])
                    if not items:
                        break 
                        
                    for item in items:
                        homepage = item.get("homepage")
                        if homepage and homepage.startswith("http") and "github.com" not in homepage:
                            fresh_urls.add(homepage.strip())
                elif response.status_code == 403:
                    time.sleep(15) # Handle rate limit
                else:
                    break
                    
                time.sleep(6) # Wait between pages
            except Exception:
                pass
            
            if len(fresh_urls) >= MAX_URLS:
                return list(fresh_urls)[:MAX_URLS]
                
    return list(fresh_urls)[:MAX_URLS]

def capture_screenshots(urls):
    """Bulletproof Playwright scraper that waits for full rendering."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"\nStarting Playwright to capture fully-rendered screenshots...")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        # Use a modern User-Agent to prevent basic bot-blocking
        context = browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
        )
        page = context.new_page()
        
        success_count = 0
        for url in tqdm(urls, desc="Snapping HD UIs"):
            try:
                domain = urlparse(url).netloc.replace("www.", "")
                safe_name = "".join(c for c in domain if c.isalnum() or c in ".-_")
                output_path = os.path.join(OUTPUT_DIR, f"{safe_name}_{abs(hash(url)) % 10000}.png")
                
                if os.path.exists(output_path):
                    continue
                
                # CRITICAL FIX 1: Wait until the network is idle (API calls finished)
                response = page.goto(url, timeout=25000, wait_until="networkidle")
                
                # CRITICAL FIX 2: Skip 404s and 500s
                if not response or not response.ok:
                    continue

                # CRITICAL FIX 3: Hard wait for 3 seconds to let React/Vue animations paint
                page.wait_for_timeout(3000)
                
                # Remove popups
                page.evaluate("""
                    document.querySelectorAll('[id*="cookie"], [class*="cookie"], [id*="popup"], [id*="banner"]').forEach(el => el.remove());
                """)
                
                page.screenshot(path=output_path)
                
                # CRITICAL FIX 4: Delete the image if it is too small (blank white/black screen)
                file_size_kb = os.path.getsize(output_path) / 1024
                if file_size_kb < MIN_FILE_SIZE_KB:
                    os.remove(output_path)
                    continue # Skip counting this as a success
                    
                success_count += 1
                
            except Exception:
                # Silently skip timeouts or sites with heavy anti-bot protection
                pass
                
        browser.close()
    print(f"\n✅ Finished! Captured {success_count} perfect HD screenshots to {OUTPUT_DIR}/")

if __name__ == "__main__":
    target_urls = get_fresh_github_urls()
    if target_urls:
        capture_screenshots(target_urls)