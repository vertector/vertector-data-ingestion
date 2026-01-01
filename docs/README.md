# Vertector Data Ingestion - Documentation

Production-grade universal multimodal data ingestion pipeline for RAG applications.

## Getting Started

- **[QuickStart Guide](quickstart.md)** - Get up and running in 5 minutes
- **[Installation Guide](installation.md)** - Complete installation instructions
- **[User Guide](user-guide.md)** - Comprehensive feature documentation
- **[Configuration Guide](configuration.md)** - Detailed configuration reference
- **[API Reference](api-reference.md)** - Complete API documentation

## What is Vertector?

Vertector is a comprehensive data ingestion system that provides:

- **Multimodal Support**: Documents (PDF, DOCX, PPTX, XLSX), Images, and Audio
- **Hardware Acceleration**: Auto-detect and optimize for MPS, CUDA, or CPU
- **Dual Pipelines**: Classic (fast) and VLM (AI-powered) processing
- **RAG-Ready**: Hierarchical chunking with metadata enrichment
- **Production Features**: Caching, retry logic, batch processing, monitoring

## Quick Examples

### Document Processing

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
result = transcriber.transcribe("meeting.wav")
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

results = vector_store.search("quantum computing", top_k=5)
```

## Documentation Structure

### For New Users

1. Start with [QuickStart Guide](quickstart.md)
2. Follow [Installation Guide](installation.md) for setup
3. Read [User Guide](user-guide.md) for features

### For Advanced Users

1. Review [Configuration Guide](configuration.md) for optimization
2. Check [API Reference](api-reference.md) for detailed API
3. Browse `../examples/` for code samples

### By Feature

- **Documents**: [User Guide - Document Processing](user-guide.md#document-processing)
- **Audio**: [User Guide - Audio Transcription](user-guide.md#audio-transcription)
- **RAG**: [User Guide - RAG Pipeline](user-guide.md#rag-pipeline)
- **Hardware**: [User Guide - Hardware Acceleration](user-guide.md#hardware-acceleration)
- **Config**: [Configuration Guide](configuration.md)

## Key Features

### Multimodal Support

| Modality | Formats | Features |
|----------|---------|----------|
| Documents | PDF, DOCX, PPTX, XLSX, HTML | Text, tables, images, OCR |
| Images | PNG, JPG | VLM understanding |
| Audio | MP3, WAV, M4A | Whisper transcription |

### Hardware Optimization

| Hardware | Acceleration | Best For |
|----------|-------------|----------|
| Apple Silicon | MLX | Local development |
| NVIDIA GPU | CUDA | Cloud deployment |
| CPU | Optimized | Universal support |

### Processing Pipelines

| Pipeline | Speed | Accuracy | Use Case |
|----------|-------|----------|----------|
| Classic | Fast | Good | Simple documents |
| VLM | Slower | Excellent | Complex layouts, images |

## Configuration Presets

Choose based on your environment:

```python
from vertector_data_ingestion import (
    LocalMpsConfig,   # Apple Silicon
    CloudGpuConfig,   # NVIDIA GPU
    CloudCpuConfig,   # CPU-only
)
```

## Support

- **Email**: cutetetteh@gmail.com
- **Issues**: Check [Troubleshooting](user-guide.md#troubleshooting)
- **Examples**: See `../examples/` directory

## License

[Your License Here]

## Documentation Index

### Guides

- [QuickStart](quickstart.md) - Get started in 5 minutes
- [Installation](installation.md) - Setup and installation
- [User Guide](user-guide.md) - Complete feature guide
- [Configuration](configuration.md) - Configuration reference

### Reference

- [API Reference](api-reference.md) - Complete API documentation

### Resources

- [Examples](../examples/) - Code examples
- [Tests](../tests/) - Test files for reference
