# Utility Scripts

Helpful scripts for common tasks with Vertector Data Ingestion.

## Available Scripts

### batch_convert.py

Batch convert documents to specified format.

**Usage**:
```bash
# Basic usage
uv run python scripts/batch_convert.py input_dir/ output_dir/

# Convert PDFs to markdown
uv run python scripts/batch_convert.py \
    documents/ \
    output/ \
    --format markdown \
    --pattern "*.pdf"

# Recursive conversion with GPU preset
uv run python scripts/batch_convert.py \
    documents/ \
    output/ \
    --format json \
    --preset cloud-gpu \
    --recursive

# Custom workers
uv run python scripts/batch_convert.py \
    documents/ \
    output/ \
    --workers 16 \
    --log-level DEBUG
```

**Options**:
- `input_dir`: Input directory containing documents
- `output_dir`: Output directory for converted files
- `--format`: Output format (markdown, json, doctags)
- `--pattern`: File pattern to match (default: `*.*`)
- `--preset`: Configuration preset (local-mps, cloud-gpu, cloud-cpu, default)
- `--workers`: Number of parallel workers
- `--log-level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `--recursive`: Process directories recursively

**Examples**:

Convert all PDFs:
```bash
uv run python scripts/batch_convert.py docs/ output/ --pattern "*.pdf"
```

Convert with Apple Silicon optimization:
```bash
uv run python scripts/batch_convert.py docs/ output/ --preset local-mps
```

Recursive conversion preserving structure:
```bash
uv run python scripts/batch_convert.py docs/ output/ --recursive
```

## Creating Custom Scripts

You can create custom scripts using Vertector as a library:

```python
#!/usr/bin/env python3
"""Custom processing script."""

from pathlib import Path
from vertector_data_ingestion import UniversalConverter, LocalMpsConfig

def main():
    config = LocalMpsConfig()
    converter = UniversalConverter(config)

    # Your custom processing logic
    for doc_path in Path("input/").glob("*.pdf"):
        doc = converter.convert_single(doc_path)
        # Process doc...

if __name__ == "__main__":
    main()
```

Make it executable:
```bash
chmod +x scripts/your_script.py
```

## See Also

- [Examples](../examples/) - Example code
- [User Guide](../docs/user-guide.md) - Feature documentation
- [Configuration Guide](../docs/configuration.md) - Configuration options
