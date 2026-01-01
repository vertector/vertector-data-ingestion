# Vertector Data Ingestion - Examples

Practical examples demonstrating key features and use cases.

## Available Examples

### 1. basic_usage.py

Basic document processing workflow.

**Features**:
- Document conversion
- Export to multiple formats
- RAG chunking
- Vector store integration
- Hardware detection

**Run**:
```bash
uv run python examples/basic_usage.py
```

### 2. audio_transcription.py

Audio transcription with Whisper.

**Features**:
- Basic transcription
- Timestamped segments
- Configuration options
- Batch processing
- SRT subtitle generation
- Multiple accuracy levels

**Run**:
```bash
uv run python examples/audio_transcription.py
```

### 3. rag_pipeline.py

Complete RAG pipeline implementation.

**Features**:
- Document to chunks workflow
- Vector store integration
- Advanced search strategies
- Batch document processing
- Metadata filtering
- Export and inspection

**Run**:
```bash
uv run python examples/rag_pipeline.py
```

## Quick Examples

### Basic Document Conversion

```python
from vertector_data_ingestion import UniversalConverter

converter = UniversalConverter()
doc = converter.convert_single("document.pdf")
print(doc.export_to_markdown())
```

### Audio Transcription

```python
from vertector_data_ingestion import create_audio_transcriber, AudioConfig

transcriber = create_audio_transcriber(AudioConfig())
result = transcriber.transcribe("audio.wav")
print(result.text)
```

### RAG Pipeline

```python
from vertector_data_ingestion import (
    UniversalConverter,
    HybridChunker,
    ChromaAdapter,
)

converter = UniversalConverter()
doc = converter.convert_single("paper.pdf")

chunker = HybridChunker()
chunks = chunker.chunk_document(doc.document)

vector_store = ChromaAdapter(collection_name="research")
vector_store.add_chunks(chunks.chunks)
```

## Common Use Cases

### 1. Research Paper Processing

```python
from pathlib import Path
from vertector_data_ingestion import (
    UniversalConverter,
    LocalMpsConfig,
    HybridChunker,
    ChromaAdapter,
)

# Configure for academic papers
config = LocalMpsConfig()
config.chunking.max_tokens = 512
config.chunking.respect_boundaries = True

converter = UniversalConverter(config)

# Process papers
papers = list(Path("papers/").glob("*.pdf"))
for paper in papers:
    doc = converter.convert_single(paper)
    chunks = HybridChunker().chunk_document(doc.document)

    # Store with metadata
    store = ChromaAdapter(collection_name="research_papers")
    store.add_chunks(chunks.chunks)
```

### 2. Meeting Transcription

```python
from pathlib import Path
from vertector_data_ingestion import (
    create_audio_transcriber,
    AudioConfig,
    WhisperModelSize,
)

# High accuracy for meetings
config = AudioConfig(
    model_size=WhisperModelSize.SMALL,
    language="en",
    word_timestamps=True,
    initial_prompt="Meeting discussion",
    beam_size=8,
)

transcriber = create_audio_transcriber(config)

# Transcribe with timestamps
result = transcriber.transcribe(Path("meeting.wav"))

# Save with formatting
with open("meeting_transcript.txt", "w") as f:
    for segment in result.segments:
        timestamp = f"[{segment.start:.1f}s]"
        f.write(f"{timestamp} {segment.text}\n")
```

### 3. Multi-Format Document Search

```python
from pathlib import Path
from vertector_data_ingestion import (
    UniversalConverter,
    HybridChunker,
    ChromaAdapter,
)

converter = UniversalConverter()
chunker = HybridChunker()
store = ChromaAdapter(collection_name="documents")

# Process different formats
formats = ["*.pdf", "*.docx", "*.pptx", "*.xlsx"]
for pattern in formats:
    for doc_path in Path("documents/").glob(pattern):
        doc = converter.convert_single(doc_path)
        chunks = chunker.chunk_document(doc.document)

        # Add format metadata
        for chunk in chunks.chunks:
            chunk.metadata["format"] = doc_path.suffix

        store.add_chunks(chunks.chunks)

# Search across all formats
results = store.search("quarterly revenue", top_k=10)
```

### 4. Batch Processing Pipeline

```python
from pathlib import Path
from vertector_data_ingestion import (
    UniversalConverter,
    CloudGpuConfig,
    ExportFormat,
)

# GPU-optimized config
config = CloudGpuConfig()
config.batch_processing_workers = 16

converter = UniversalConverter(config)

# Batch convert
documents = list(Path("input/").glob("*.pdf"))
results = converter.convert_batch(documents)

# Export all
output_dir = Path("output/")
output_dir.mkdir(exist_ok=True)

for doc in results:
    output_path = output_dir / f"{doc.metadata.source_path.stem}.md"
    converter.convert_and_export(
        source=doc.metadata.source_path,
        output_path=output_path,
        format=ExportFormat.MARKDOWN
    )
```

## Testing Examples

To test examples without actual files:

```bash
# Shows configurations without requiring files
uv run python examples/audio_transcription.py

# Demonstrates RAG concepts
uv run python examples/rag_pipeline.py
```

## Hardware-Specific Examples

### Apple Silicon (MLX)

```python
from vertector_data_ingestion import LocalMpsConfig, UniversalConverter

config = LocalMpsConfig()  # Auto-optimized for MPS
converter = UniversalConverter(config)
```

### NVIDIA GPU

```python
from vertector_data_ingestion import CloudGpuConfig, UniversalConverter

config = CloudGpuConfig()  # Auto-optimized for CUDA
converter = UniversalConverter(config)
```

### CPU-Only

```python
from vertector_data_ingestion import CloudCpuConfig, UniversalConverter

config = CloudCpuConfig()  # CPU-optimized
converter = UniversalConverter(config)
```

## Next Steps

- Read [QuickStart Guide](../docs/quickstart.md) for basics
- Review [User Guide](../docs/user-guide.md) for features
- Check [Configuration Guide](../docs/configuration.md) for options
- See [API Reference](../docs/api-reference.md) for details

## Getting Help

Contact: cutetetteh@gmail.com
