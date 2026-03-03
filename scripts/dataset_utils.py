"""
Utility Scripts for Data Factory

1. merge_datasets.py - Merge synthetic dataset with existing dataset
2. validate_sql.py - Validate SQL syntax in dataset
"""

import json
import argparse
from pathlib import Path
import sqlparse
from typing import List, Dict, Any

def validate_sql(sql_string: str) -> tuple[bool, str]:
    """
    Validate SQL syntax.
    Returns (is_valid, error_message)
    """
    try:
        # Parse SQL
        parsed = sqlparse.parse(sql_string)
        if not parsed:
            return False, "Empty SQL"
        
        # Check for basic syntax
        for statement in parsed:
            if not statement.get_type():
                continue
            # If sqlparse can parse it without error, it's likely valid
        
        return True, ""
    except Exception as e:
        return False, str(e)


def merge_datasets(
    synthetic_path: str,
    existing_path: str,
    output_path: str,
    validate: bool = True
) -> int:
    """
    Merge synthetic dataset with existing dataset.
    Returns count of total items.
    """
    print(f"\n📊 Merging datasets...")
    
    # Load synthetic data
    with open(synthetic_path, 'r') as f:
        synthetic = json.load(f)
    print(f"  Loaded {len(synthetic)} synthetic items")
    
    # Load existing data
    existing = []
    if Path(existing_path).exists():
        with open(existing_path, 'r') as f:
            existing = json.load(f)
        print(f"  Loaded {len(existing)} existing items")
    
    # Merge (synthetic first, then existing)
    merged = synthetic + existing
    
    # Validate SQL if requested
    if validate:
        print(f"  Validating SQL...")
        valid_count = 0
        invalid_items = []
        
        for i, item in enumerate(merged):
            sql = item.get('output', '')
            is_valid, error = validate_sql(sql)
            
            if is_valid:
                valid_count += 1
            else:
                invalid_items.append({
                    'index': i,
                    'image': item.get('image_path', 'unknown'),
                    'error': error
                })
        
        print(f"  ✅ {valid_count}/{len(merged)} items have valid SQL")
        
        if invalid_items:
            print(f"  ⚠️  {len(invalid_items)} items have invalid SQL")
            for item in invalid_items[:5]:  # Show first 5
                print(f"     - {item['image']}: {item['error'][:50]}")
            if len(invalid_items) > 5:
                print(f"     ... and {len(invalid_items) - 5} more")
    
    # Save merged
    with open(output_path, 'w') as f:
        json.dump(merged, f, indent=2)
    
    print(f"  📁 Saved {len(merged)} items to {output_path}")
    return len(merged)


def validate_dataset(dataset_path: str) -> None:
    """
    Validate all SQL statements in a dataset.
    """
    print(f"\n🔍 Validating dataset: {dataset_path}")
    
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)
    
    valid = 0
    invalid = 0
    errors = {}
    
    for i, item in enumerate(dataset):
        sql = item.get('output', '')
        is_valid, error = validate_sql(sql)
        
        if is_valid:
            valid += 1
        else:
            invalid += 1
            if error not in errors:
                errors[error] = 0
            errors[error] += 1
    
    print(f"\n  Results:")
    print(f"  ✅ Valid:   {valid}/{len(dataset)}")
    print(f"  ❌ Invalid: {invalid}/{len(dataset)}")
    
    if errors:
        print(f"\n  Error breakdown:")
        for error, count in sorted(errors.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"    - {error[:60]}: {count} items")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data Factory Utilities")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Merge command
    merge_parser = subparsers.add_parser("merge", help="Merge datasets")
    merge_parser.add_argument("--synthetic", required=True, help="Synthetic dataset path")
    merge_parser.add_argument("--existing", required=True, help="Existing dataset path")
    merge_parser.add_argument("--output", required=True, help="Output dataset path")
    merge_parser.add_argument("--no-validate", action="store_true", help="Skip validation")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dataset")
    validate_parser.add_argument("--dataset", required=True, help="Dataset path")
    
    args = parser.parse_args()
    
    if args.command == "merge":
        merge_datasets(
            args.synthetic,
            args.existing,
            args.output,
            validate=not args.no_validate
        )
    elif args.command == "validate":
        validate_dataset(args.dataset)
    else:
        parser.print_help()
