#!/usr/bin/env python3
"""
Test suite for Data Factory - verify all components work before scaling to 5000
"""

import os
import json
import sys
from pathlib import Path
from dotenv import load_dotenv

def test_imports():
    """Test that all required dependencies are installed."""
    print("\n🔧 TESTING IMPORTS...")
    
    missing = []
    
    dependencies = {
        "google.generativeai": "google-generativeai",
        "PIL": "pillow",
        "jinja2": "jinja2",
        "sqlparse": "sqlparse",
        "dotenv": "python-dotenv",
    }
    
    for module, package in dependencies.items():
        try:
            __import__(module)
            print(f"  ✅ {module}")
        except ImportError:
            print(f"  ❌ {module}")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print(f"   Run: pip install {' '.join(missing)}")
        return False
    
    print("  ✅ All imports OK")
    return True


def test_gemini_api():
    """Test Gemini API connectivity and key validity."""
    print("\n🌐 TESTING GEMINI API...")
    
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    
    if not api_key:
        print("  ❌ GEMINI_API_KEY not found in environment")
        return False
    
    print(f"  ✅ API key found: {api_key[:20]}...")
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        
        # Try a simple request
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content("Say 'hello'")
        
        if response.text:
            print(f"  ✅ API working: {response.text[:30]}")
            return True
        else:
            print("  ❌ API returned empty response")
            return False
            
    except Exception as e:
        print(f"  ❌ API error: {str(e)[:100]}")
        return False


def test_jinja_rendering():
    """Test Jinja2 HTML rendering."""
    print("\n🎨 TESTING JINJA2 RENDERING...")
    
    try:
        from jinja2 import Template
        
        template_str = """
        <h1>{{ name }}</h1>
        <p>{{ description }}</p>
        """
        
        template = Template(template_str)
        result = template.render(name="Test", description="This is a test")
        
        if "Test" in result and "test" in result:
            print(f"  ✅ Jinja2 rendering works")
            return True
        else:
            print(f"  ❌ Rendering failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_playwright_installation():
    """Test Playwright installation (without launching browser)."""
    print("\n📸 TESTING PLAYWRIGHT...")
    
    try:
        # Check if playwright is installed
        import playwright
        print(f"  ✅ Playwright module found")
        
        # Check if browsers are installed
        browsers_dir = Path.home() / ".cache" / "ms-playwright"
        if browsers_dir.exists():
            print(f"  ✅ Browsers cached")
            return True
        else:
            print(f"  ⚠️  Browsers not cached. Run: playwright install")
            return True  # Module is installed, just needs browser download
            
    except ImportError:
        print(f"  ❌ Playwright not installed")
        print(f"     Run: pip install playwright && playwright install")
        return False
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_sql_parsing():
    """Test SQL parsing capability."""
    print("\n🗄️  TESTING SQL PARSING...")
    
    try:
        import sqlparse
        
        test_sql = """
        CREATE TABLE users (
            id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            email VARCHAR(100) UNIQUE
        );
        """
        
        parsed = sqlparse.parse(test_sql)
        if parsed and len(parsed) > 0:
            print(f"  ✅ SQL parsing works")
            return True
        else:
            print(f"  ❌ Parsing failed")
            return False
            
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")
        return False


def test_data_structure():
    """Test that data directories exist."""
    print("\n📁 TESTING DATA STRUCTURE...")
    
    required_dirs = [
        "data",
        "data/ui_screenshots",
        "scripts",
    ]
    
    missing = []
    for dir_path in required_dirs:
        if Path(dir_path).exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path}")
            missing.append(dir_path)
    
    if missing:
        print(f"\n   Creating missing directories...")
        for dir_path in missing:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            print(f"   ✅ Created {dir_path}")
    
    return True


def test_requirements_file():
    """Check requirements.txt has necessary dependencies."""
    print("\n📦 CHECKING REQUIREMENTS.TXT...")
    
    with open("requirements.txt", "r") as f:
        content = f.read()
    
    required = ["playwright", "jinja2", "google-generativeai"]
    missing = []
    
    for pkg in required:
        if pkg.lower() in content.lower():
            print(f"  ✅ {pkg} in requirements.txt")
        else:
            print(f"  ❌ {pkg} NOT in requirements.txt")
            missing.append(pkg)
    
    if missing:
        print(f"\n   ⚠️  Add these to requirements.txt: {', '.join(missing)}")
    
    return len(missing) == 0


def run_all_tests():
    """Run all tests and report results."""
    print("\n" + "="*80)
    print("🧪 DATA FACTORY TEST SUITE")
    print("="*80)
    
    tests = [
        ("Imports", test_imports),
        ("Requirements.txt", test_requirements_file),
        ("Data Structure", test_data_structure),
        ("SQL Parsing", test_sql_parsing),
        ("Jinja2 Rendering", test_jinja_rendering),
        ("Playwright", test_playwright_installation),
        ("Gemini API", test_gemini_api),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"\n  ❌ Test crashed: {str(e)[:100]}")
            results[name] = False
    
    # Summary
    print("\n" + "="*80)
    print("📊 TEST RESULTS")
    print("="*80)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} passed")
    
    if passed == total:
        print("\n✅ ALL TESTS PASSED! Ready to run data factory.")
        print("\nNext steps:")
        print("  1. Test with small batch: python scripts/data_factory.py 10")
        print("  2. If successful, scale up: python scripts/data_factory.py 5000")
        return 0
    else:
        print("\n❌ Some tests failed. Fix issues above before running data factory.")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
