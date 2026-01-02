# Embedding Model Configuration

The system supports any Hugging Face compatible embedding model and automatically detects the appropriate tokenizer configuration.

## Quick Start

### Using Default (Qwen3)

```python
from vertector_data_ingestion import HybridChunker, ChromaAdapter

# Uses Qwen/Qwen3-Embedding-0.6B by default
chunker = HybridChunker()
vector_store = ChromaAdapter(collection_name="my_collection")
```

### Using Custom Model

```python
from vertector_data_ingestion import HybridChunker, ChromaAdapter
from vertector_data_ingestion.models.config import ChunkingConfig

# Configure with different model
chunk_config = ChunkingConfig(
    tokenizer="sentence-transformers/all-MiniLM-L6-v2",
    max_tokens=512,
)

chunker = HybridChunker(config=chunk_config)
vector_store = ChromaAdapter(
    collection_name="my_collection",
    embedding_model="sentence-transformers/all-MiniLM-L6-v2"
)
```

### Using Environment Variables

```bash
export VERTECTOR_CHUNK_TOKENIZER="intfloat/e5-large-v2"
export VERTECTOR_CHUNK_MAX_TOKENS=512
export VERTECTOR_EMBEDDING_MODEL="intfloat/e5-large-v2"
```

Then in Python:
```python
# Reads from environment variables
chunker = HybridChunker()
vector_store = ChromaAdapter(collection_name="my_collection")
```

## Automatic Padding Detection

The system automatically detects whether a model needs left or right padding:

- **Qwen models** → left padding (auto-detected)
- **All other models** → right padding (default)

This happens automatically based on the model name, so you don't need to worry about it.

## Recommended Models

### High Quality (Best Accuracy)

**Qwen/Qwen3-Embedding-0.6B** (default)
- Context: 32,768 tokens
- Size: ~4GB
- Best for: Maximum quality, long documents
- Padding: Left (auto-detected)

```python
chunk_config = ChunkingConfig(
    tokenizer="Qwen/Qwen3-Embedding-0.6B",
    max_tokens=8192,  # Can go up to 32,768
)
```

**BAAI/bge-large-en-v1.5**
- Context: 512 tokens
- Size: ~1.3GB
- Best for: English documents, high quality
- Padding: Right (auto-detected)

```python
chunk_config = ChunkingConfig(
    tokenizer="BAAI/bge-large-en-v1.5",
    max_tokens=512,
)
```

### Balanced (Good Quality + Speed)

**intfloat/e5-large-v2**
- Context: 512 tokens
- Size: ~1.3GB
- Best for: General purpose, balanced performance
- Padding: Right (auto-detected)

```python
chunk_config = ChunkingConfig(
    tokenizer="intfloat/e5-large-v2",
    max_tokens=512,
)
```

### Fast (Best Speed)

**sentence-transformers/all-MiniLM-L6-v2**
- Context: 512 tokens
- Size: ~90MB
- Best for: Speed, resource-constrained environments
- Padding: Right (auto-detected)

```python
chunk_config = ChunkingConfig(
    tokenizer="sentence-transformers/all-MiniLM-L6-v2",
    max_tokens=512,
)
```

### Multilingual

**BAAI/bge-m3**
- Context: 8,192 tokens
- Size: ~2.3GB
- Languages: 100+
- Best for: Non-English or mixed language documents
- Padding: Right (auto-detected)

```python
chunk_config = ChunkingConfig(
    tokenizer="BAAI/bge-m3",
    max_tokens=8192,
)
```

## Model Requirements

### For Chunking (Tokenizer)
- Must be compatible with Hugging Face `AutoTokenizer`
- Examples: Any model on Hugging Face with a tokenizer

### For Embeddings (Vector Store)
- Must be compatible with `SentenceTransformers` or Hugging Face `AutoModel`
- Must support `.encode()` for generating embeddings
- Examples: Most embedding models on Hugging Face

## Important Notes

1. **Use the same model for chunking and embeddings**
   - The tokenizer should match the embedding model
   - This ensures proper token counting and retrieval

2. **Respect max_tokens limits**
   - Each model has a maximum context window
   - Set `max_tokens` to match your model's capacity

3. **Consider your use case**
   - Long documents → Use models with large context (Qwen3, BGE-M3)
   - Speed critical → Use smaller models (MiniLM)
   - Multilingual → Use multilingual models (BGE-M3)

## Complete Example

See `examples/custom_embedding_models.py` for complete working examples with different models.

```python
from pathlib import Path
from vertector_data_ingestion import (
    UniversalConverter,
    HybridChunker,
    ChromaAdapter,
    LocalMpsConfig,
)
from vertector_data_ingestion.models.config import ChunkingConfig

# Configure with E5 large
chunk_config = ChunkingConfig(
    tokenizer="intfloat/e5-large-v2",
    max_tokens=512,
)

# Create pipeline
converter = UniversalConverter(LocalMpsConfig())
chunker = HybridChunker(config=chunk_config)
vector_store = ChromaAdapter(
    collection_name="e5_docs",
    embedding_model="intfloat/e5-large-v2",
)

# Process document
doc = converter.convert(Path("document.pdf"))
chunks = chunker.chunk_document(doc)
vector_store.add_chunks(chunks.chunks)

# Search
results = vector_store.search("your query", top_k=5)
```

## Troubleshooting

### Import errors
```python
# If model not found, it will download automatically
# Make sure you have internet connection on first use
```

### Memory issues
```python
# Use smaller models if you have limited RAM
chunk_config = ChunkingConfig(
    tokenizer="sentence-transformers/all-MiniLM-L6-v2",  # Only ~90MB
    max_tokens=512,
)
```

### Slow performance
```python
# Use GPU if available, or switch to smaller models
# Check hardware with:
from vertector_data_ingestion import HardwareDetector
hw_info = HardwareDetector.get_device_info()
print(hw_info)
```
