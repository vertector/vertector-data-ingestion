# QuickStart Guide

Get started with Vertector Data Ingestion in 5 minutes.

## What is Vertector?

Vertector is a production-grade multimodal data ingestion pipeline that converts documents, images, and audio into RAG-ready formats. It automatically detects your hardware (Apple Silicon, NVIDIA GPU, or CPU) and optimizes processing accordingly.

## Installation

```bash
# Clone and install
git clone <repository-url>
cd vertector-data-ingestion
uv sync
```

## Your First Document

Convert a PDF to Markdown in 3 lines:

```python
from vertector_data_ingestion import UniversalConverter

converter = UniversalConverter()
doc = converter.convert("document.pdf")
print(doc.export_to_markdown())
```

## Common Use Cases

### 1. Document Processing

```python
from pathlib import Path
from vertector_data_ingestion import UniversalConverter, ExportFormat

converter = UniversalConverter()

# Convert PDF, DOCX, PPTX, or XLSX
doc = converter.convert(Path("presentation.pptx"))

# Export to different formats
markdown = converter.export(doc, ExportFormat.MARKDOWN)
json_output = converter.export(doc, ExportFormat.JSON)

# Save directly to file (auto-creates output directory)
# Just provide filename - saves to configured output_dir
output_file = converter.convert_and_export(
    source=Path("document.pdf"),
    output_name="mydoc.md",  # Saved to ./output/mydoc.md
    format=ExportFormat.MARKDOWN
)

# Or let it auto-generate filename from source
output_file = converter.convert_and_export(
    source=Path("document.pdf")  # Saved to ./output/document.md
)
```

### 2. Audio Transcription

```python
from pathlib import Path
from vertector_data_ingestion import (
    create_audio_transcriber,
    AudioConfig,
    WhisperModelSize,
)

# Create transcriber with config
config = AudioConfig(
    model_size=WhisperModelSize.BASE,
    language="en",
    word_timestamps=True,
)

transcriber = create_audio_transcriber(config)

# Transcribe audio
result = transcriber.transcribe(Path("meeting.wav"))

print(f"Text: {result.text}")
print(f"Language: {result.language}")
print(f"Duration: {result.duration:.2f}s")

# Access timestamped segments
for segment in result.segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s]: {segment.text}")
```

### 3. RAG Pipeline

```python
from vertector_data_ingestion import (
    UniversalConverter,
    HybridChunker,
    ChromaAdapter,
)

# Convert document
converter = UniversalConverter()
doc = converter.convert_single("research_paper.pdf")

# Create chunks optimized for RAG
chunker = HybridChunker()
chunks = chunker.chunk_document(doc.document)

# Store in vector database
vector_store = ChromaAdapter(collection_name="research")
vector_store.add_chunks(chunks)

# Search
results = vector_store.search("quantum computing", top_k=5)
for result in results:
    print(f"Page {result['metadata']['page_no']}: {result['text'][:100]}...")
```

### 4. Batch Processing

```python
from pathlib import Path
from vertector_data_ingestion import UniversalConverter

converter = UniversalConverter()

# Process multiple files - just pass a list!
documents = [
    Path("doc1.pdf"),
    Path("doc2.docx"),
    Path("doc3.pptx"),
]

# Same convert() method handles both single and batch
results = converter.convert(documents)

for doc in results:
    print(f"Processed: {doc.metadata.source_path}")
    print(f"Pages: {doc.metadata.num_pages}")
    print(f"Time: {doc.metadata.processing_time:.2f}s\n")
```

## Hardware Optimization

Vertector automatically detects and uses the best hardware available:

### Apple Silicon (M1/M2/M3)

```python
from vertector_data_ingestion import LocalMpsConfig, UniversalConverter

# Optimized for Apple Silicon with MLX acceleration
config = LocalMpsConfig()
converter = UniversalConverter(config)

doc = converter.convert("document.pdf")
```

### NVIDIA GPU

```python
from vertector_data_ingestion import CloudGpuConfig, UniversalConverter

# Optimized for CUDA acceleration
config = CloudGpuConfig()
converter = UniversalConverter(config)

doc = converter.convert("document.pdf")
```

### CPU-Only

```python
from vertector_data_ingestion import CloudCpuConfig, UniversalConverter

# CPU-optimized settings
config = CloudCpuConfig()
converter = UniversalConverter(config)

doc = converter.convert("document.pdf")
```

## Configuration Presets

Choose a preset based on your environment:

| Preset | Hardware | Use Case |
|--------|----------|----------|
| `LocalMpsConfig` | Apple Silicon | Local development on Mac |
| `CloudGpuConfig` | NVIDIA GPU | Cloud deployment with GPU |
| `CloudCpuConfig` | CPU only | Cloud deployment without GPU |

## Environment Variables

Quick configuration via environment variables:

```bash
# General settings
export VERTECTOR_LOG_LEVEL=INFO
export VERTECTOR_CACHE_DIR=/path/to/cache

# VLM settings (for image/document understanding)
export VERTECTOR_VLM_USE_MLX=true
export VERTECTOR_VLM_PRESET_MODEL=granite-mlx
export VERTECTOR_VLM_BATCH_SIZE=8

# Audio settings
export VERTECTOR_AUDIO_MODEL_SIZE=base
export VERTECTOR_AUDIO_BACKEND=mlx
export VERTECTOR_AUDIO_LANGUAGE=en

# OCR settings
export VERTECTOR_OCR_ENGINE=easyocr
export VERTECTOR_OCR_USE_GPU=true
```

## Next Steps

- Read the [User Guide](user-guide.md) for comprehensive feature documentation
- Check [Configuration](configuration.md) for advanced customization
- Browse [examples/](../examples/) for more code samples
- See [API Reference](api-reference.md) for detailed API documentation

## Common Issues

**Import Error**: Make sure you've installed the package:
```bash
uv sync
```

**MLX not available**: Install MLX Whisper for Apple Silicon:
```bash
uv add mlx-whisper
```

**CUDA out of memory**: Reduce batch size:
```python
config = CloudGpuConfig()
config.vlm.batch_size = 4  # Reduce from default
```

## Getting Help

- Check [Troubleshooting](user-guide.md#troubleshooting) section
- Review [Known Limitations](user-guide.md#known-limitations)
- Contact: cutetetteh@gmail.com
