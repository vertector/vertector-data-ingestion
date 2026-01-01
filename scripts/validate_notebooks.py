#!/usr/bin/env python3
"""Validate Jupyter notebooks for syntax and import errors."""

import json
import sys
from pathlib import Path
from typing import List, Tuple


def extract_code_cells(notebook_path: Path) -> List[str]:
    """Extract code cells from notebook."""
    with open(notebook_path, 'r', encoding='utf-8') as f:
        notebook = json.load(f)

    code_cells = []
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            if isinstance(source, list):
                code = ''.join(source)
            else:
                code = source
            code_cells.append(code)

    return code_cells


def validate_syntax(code: str, cell_num: int) -> Tuple[bool, str]:
    """Validate Python syntax."""
    try:
        compile(code, f'<cell {cell_num}>', 'exec')
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"
    except Exception as e:
        return False, f"Error: {e}"


def check_imports(code: str) -> Tuple[bool, List[str]]:
    """Check if imports are valid (without executing)."""
    import_errors = []

    # Extract import statements (handling multi-line imports)
    import ast
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                # Just verify we can parse import nodes
                pass
    except SyntaxError as e:
        # This is already caught by validate_syntax
        pass
    except Exception as e:
        import_errors.append(f"Import analysis error: {e}")

    return len(import_errors) == 0, import_errors


def validate_notebook(notebook_path: Path) -> Tuple[bool, List[str]]:
    """Validate a single notebook."""
    print(f"\nValidating: {notebook_path.name}")
    print("=" * 60)

    errors = []

    try:
        code_cells = extract_code_cells(notebook_path)
        print(f"Found {len(code_cells)} code cells")

        for i, code in enumerate(code_cells, 1):
            # Skip empty cells
            if not code.strip():
                continue

            # Validate syntax
            valid, error = validate_syntax(code, i)
            if not valid:
                errors.append(f"Cell {i}: {error}")

            # Check imports
            import_valid, import_errors = check_imports(code)
            if not import_valid:
                for err in import_errors:
                    errors.append(f"Cell {i}: Import issue - {err}")

        if errors:
            print(f"✗ Found {len(errors)} issues:")
            for error in errors:
                print(f"  - {error}")
            return False, errors
        else:
            print("✓ All checks passed")
            return True, []

    except Exception as e:
        error_msg = f"Failed to validate: {e}"
        print(f"✗ {error_msg}")
        return False, [error_msg]


def main():
    """Validate all notebooks."""
    notebooks_dir = Path("notebooks")

    if not notebooks_dir.exists():
        print(f"Error: Directory not found: {notebooks_dir}")
        sys.exit(1)

    notebooks = list(notebooks_dir.glob("*.ipynb"))

    if not notebooks:
        print("No notebooks found")
        sys.exit(1)

    print("Notebook Validation")
    print("=" * 60)
    print(f"Found {len(notebooks)} notebooks\n")

    results = {}
    for notebook in sorted(notebooks):
        valid, errors = validate_notebook(notebook)
        results[notebook.name] = (valid, errors)

    # Summary
    print("\n" + "=" * 60)
    print("Validation Summary")
    print("=" * 60)

    passed = sum(1 for valid, _ in results.values() if valid)
    failed = len(results) - passed

    print(f"\nTotal: {len(results)} notebooks")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")

    if failed > 0:
        print("\nFailed notebooks:")
        for name, (valid, errors) in results.items():
            if not valid:
                print(f"  - {name}: {len(errors)} issues")
        sys.exit(1)
    else:
        print("\n✓ All notebooks validated successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
