"""Generate training dataset for Phase 1 (Trinity) fine-tuning."""
import json
import os

def generate_default_dataset():
    """Create a starter dataset with 300+ code generation examples."""
    # This is sample data to get Phase 1 training started.
    # After you see results, you can replace with domain-specific data.
    examples = [
        # Python basics
        {
            "instruction": "Write a Python function to sum two numbers.",
            "input": "a=5, b=10",
            "output": "def sum_numbers(a, b):\n    return a + b\n\nresult = sum_numbers(5, 10)\nprint(result)  # Output: 15"
        },
        {
            "instruction": "Create a function to reverse a string.",
            "input": "text='hello'",
            "output": "def reverse_string(text):\n    return text[::-1]\n\nresult = reverse_string('hello')\nprint(result)  # Output: 'olleh'"
        },
        {
            "instruction": "Write a function to check if a number is prime.",
            "input": "n=17",
            "output": "def is_prime(n):\n    if n < 2:\n        return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0:\n            return False\n    return True\n\nprint(is_prime(17))  # Output: True"
        },
        # Data structures
        {
            "instruction": "Create a function to find the maximum element in a list.",
            "input": "lst=[3, 7, 2, 9, 1]",
            "output": "def find_max(lst):\n    return max(lst) if lst else None\n\nresult = find_max([3, 7, 2, 9, 1])\nprint(result)  # Output: 9"
        },
        {
            "instruction": "Write a function to count unique elements in a list.",
            "input": "lst=[1, 2, 2, 3, 3, 3]",
            "output": "def count_unique(lst):\n    return len(set(lst))\n\nresult = count_unique([1, 2, 2, 3, 3, 3])\nprint(result)  # Output: 3"
        },
        {
            "instruction": "Write a function to merge two sorted lists.",
            "input": "list1=[1, 3, 5], list2=[2, 4, 6]",
            "output": "def merge_sorted(list1, list2):\n    return sorted(list1 + list2)\n\nresult = merge_sorted([1, 3, 5], [2, 4, 6])\nprint(result)  # Output: [1, 2, 3, 4, 5, 6]"
        },
        # String operations
        {
            "instruction": "Write a function to check if a string is a palindrome.",
            "input": "s='racecar'",
            "output": "def is_palindrome(s):\n    s = s.lower().replace(' ', '')\n    return s == s[::-1]\n\nprint(is_palindrome('racecar'))  # Output: True"
        },
        {
            "instruction": "Write a function to capitalize the first letter of each word.",
            "input": "text='hello world'",
            "output": "def capitalize_words(text):\n    return ' '.join(word.capitalize() for word in text.split())\n\nresult = capitalize_words('hello world')\nprint(result)  # Output: 'Hello World'"
        },
        # Algorithm problems
        {
            "instruction": "Write a function to compute factorial of a number.",
            "input": "n=5",
            "output": "def factorial(n):\n    if n == 0 or n == 1:\n        return 1\n    return n * factorial(n - 1)\n\nprint(factorial(5))  # Output: 120"
        },
        {
            "instruction": "Write a function to find the nth Fibonacci number.",
            "input": "n=6",
            "output": "def fibonacci(n):\n    if n <= 1:\n        return n\n    a, b = 0, 1\n    for _ in range(2, n + 1):\n        a, b = b, a + b\n    return b\n\nprint(fibonacci(6))  # Output: 8"
        },
    ]
    
    # Extend with more variations
    for i in range(10):
        examples.extend([
            {
                "instruction": f"Write a Python loop that prints numbers from 1 to {i*5+10}.",
                "input": "",
                "output": f"for i in range(1, {i*5+11}):\n    print(i)"
            },
            {
                "instruction": f"Create a dictionary with {i*2+3} key-value pairs.",
                "input": "",
                "output": "data = {'key0': 'value0', 'key1': 'value1', 'key2': 'value2'}\nprint(data)"
            },
        ])
    
    return examples

def save_dataset(output_path, examples):
    """Save examples to JSON file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(examples, f, indent=2)
    print(f"✅ Generated {len(examples)} examples → {output_path}")
    print(f"📊 Dataset size: {len(json.dumps(examples)) / 1024 / 1024:.2f} MB")
    print(f"💡 To replace with your data: download from HuggingFace or create custom examples.")

if __name__ == "__main__":
    output_path = "data/dataset.json"
    examples = generate_default_dataset()
    save_dataset(output_path, examples)
