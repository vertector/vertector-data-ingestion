# Vertector Data Ingestion

<div align="center">

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![CI](https://github.com/vertector/vertector-data-ingestion/workflows/CI/badge.svg)](https://github.com/vertector/vertector-data-ingestion/actions)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Production-grade universal multimodal data ingestion pipeline for RAG applications**

[Features](#features) • [Installation](#installation) • [Quick Start](#quick-start) • [MCP Server](#mcp-server) • [Documentation](#documentation)

</div>

---

## Overview

Vertector is a comprehensive data ingestion system built on IBM's Docling framework, designed for RAG (Retrieval-Augmented Generation) applications. It provides unified processing for multiple data modalities with hardware acceleration and production-ready features.

### Why Vertector?

- **🎯 One Pipeline, All Formats**: Process documents, images, and audio with a single unified API
- **⚡ Hardware Accelerated**: Auto-detect and optimize for Apple Silicon (MPS), NVIDIA (CUDA), or CPU
- **🧠 RAG-Optimized**: Hierarchical chunking with rich metadata for better retrieval
- **🚀 Production Ready**: Caching, retry logic, batch processing, and monitoring built-in
- **🔌 MCP Integration**: Native Model Context Protocol server for AI assistants

## Features

### 📄 Multimodal Support

| Modality | Formats | Features |
|----------|---------|----------|
| **Documents** | PDF, DOCX, PPTX, XLSX, HTML | Advanced table detection, layout preservation |
| **Images** | PNG, JPG, TIFF | Vision-Language Model processing |
| **Audio** | WAV, MP3, M4A, FLAC | Speech-to-text with Whisper (MLX/CUDA/CPU) |

### 🤖 Intelligent Processing

- **Dual Pipeline Architecture**: Auto-routing between Classic (layout-based) and VLM (vision-based) pipelines
- **Hardware Auto-Detection**: Optimize for MPS (Apple Silicon), CUDA (NVIDIA GPU), or CPU
- **OCR Engine Selection**: EasyOCR (GPU-accelerated), Tesseract (lightweight), OCRMac (native macOS)
- **VLM Model Presets**: Granite, SmolDocling, Qwen2.5-VL, Pixtral, Gemma3, Phi4

### 🎯 RAG-Ready Output

- **Hierarchical Chunking**: Token-aware chunking with section hierarchy and metadata
- **Multiple Export Formats**: Markdown, JSON, DocTags
- **Vector Store Integration**: ChromaDB, Pinecone, Qdrant, OpenSearch
- **Embedding Support**: Qwen3-Embedding-0.6B (32K context window)

### 🔌 MCP Server

Native Model Context Protocol server with 14 tools for AI assistants:

- **Document Processing** (5 tools): convert, batch convert, extract metadata/tables/images
- **Chunking** (3 tools): chunk documents/text, analyze distribution
- **Audio** (2 tools): transcribe audio, batch transcribe
- **Utilities** (4 tools): detect hardware, list formats, validate files, estimate time

[Learn more about MCP Server →](docs/mcp-server.md)

### 🚀 Production Features

- **Unified Configuration**: Environment variables, config files, or programmatic
- **Caching & Retry Logic**: Configurable caching with intelligent retry
- **Batch Processing**: Multi-worker parallel processing
- **Monitoring**: Structured logging and metrics collection

## Installation

### Prerequisites

- **Python 3.12+**
- **uv package manager** (recommended)

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Quick Install

```bash
# Clone repository
git clone https://github.com/vertector/vertector-data-ingestion.git
cd vertector-data-ingestion

# Install core dependencies
uv sync

# Install with all features
uv sync --all-extras
```

### Optional Dependencies

```bash
# Audio transcription
uv sync --extra asr         # Whisper (MLX + standard)

# Apple Silicon acceleration
uv sync --extra mlx         # MLX for M1/M2/M3/M4

# MCP server
uv sync --extra mcp         # Model Context Protocol

# Development
uv sync --extra dev         # Testing, linting, type checking
```

For detailed installation instructions, see [Installation Guide](docs/installation.md).

## Quick Start

### Basic Document Processing

```python
from pathlib import Path
from vertector_data_ingestion import UniversalConverter, ExportFormat

# Initialize with default config
converter = UniversalConverter()

# Convert single document
doc = converter.convert_single(Path("document.pdf"))

# Export to markdown
markdown = converter.export(doc, ExportFormat.MARKDOWN)
print(markdown)
```

### Hardware-Optimized Processing

```python
from vertector_data_ingestion import LocalMpsConfig, UniversalConverter

# Use optimized config for Apple Silicon
config = LocalMpsConfig()
converter = UniversalConverter(config)

# Process document with MPS acceleration
doc = converter.convert_single("presentation.pptx")
```

### Audio Transcription

```python
from pathlib import Path
from vertector_data_ingestion import (
    AudioConfig,
    WhisperModelSize,
    AudioBackend,
    create_audio_transcriber,
)

# Configure audio transcription
audio_config = AudioConfig(
    model_size=WhisperModelSize.BASE,
    backend=AudioBackend.AUTO,  # Auto-detect best backend
    language="en",
    word_timestamps=True,
)

# Create transcriber
transcriber = create_audio_transcriber(audio_config)

# Transcribe audio file
result = transcriber.transcribe(Path("meeting.wav"))

print(f"Transcription: {result.text}")
print(f"Language: {result.language}")
print(f"Duration: {result.duration:.2f}s")

# Access timestamped segments
for segment in result.segments:
    print(f"[{segment.start:.2f}s - {segment.end:.2f}s]: {segment.text}")
```

### RAG Pipeline with Chunking

```python
from vertector_data_ingestion import (
    UniversalConverter,
    HybridChunker,
    ChromaAdapter,
)

# Convert document
converter = UniversalConverter()
doc = converter.convert("research_paper.pdf")

# Create chunks for RAG
chunker = HybridChunker()
chunks = chunker.chunk_document(doc.document)

# Store in vector database
vector_store = ChromaAdapter(collection_name="research_papers")
vector_store.add_chunks(chunks)
```

For more examples, see [Quick Start Guide](docs/quickstart.md).

## MCP Server

Vertector includes a native Model Context Protocol (MCP) server that exposes all processing capabilities as tools for AI assistants like Claude.

### Installation

```bash
# Install with MCP support
uv sync --extra mcp

# Verify installation
uv run vertector-data-ingestion-mcp --help
```

### Configuration for Claude Desktop

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "vertector-data-ingestion-mcp": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/vertector-data-ingestion",
        "run",
        "vertector-data-ingestion-mcp"
      ]
    }
  }
}
```

### Available Tools

#### Document Processing
- `convert_document` - Convert documents with hardware auto-detection
- `batch_convert_documents` - Process multiple documents efficiently
- `extract_metadata` - Extract document metadata and structure
- `extract_tables` - Extract tables with formatting preservation
- `extract_images` - Extract images with bounding boxes

#### Chunking
- `chunk_document` - Create semantic chunks from documents
- `chunk_text` - Chunk raw text directly
- `analyze_chunk_distribution` - Analyze chunk size statistics

#### Audio
- `transcribe_audio` - Transcribe audio files with hardware acceleration
- `batch_transcribe_audio` - Process multiple audio files

#### Utilities
- `detect_hardware` - Detect MPS/CUDA/CPU capabilities
- `list_export_formats` - List supported export formats
- `validate_file` - Validate file format support
- `estimate_processing_time` - Estimate processing time for a file

For complete MCP documentation, see [MCP Server Guide](docs/mcp-server.md).

## Configuration

### Configuration Presets

Vertector provides three optimized configuration presets:

| Preset | Hardware | Best For | Features |
|--------|----------|----------|----------|
| `LocalMpsConfig` | Apple Silicon (M1/M2/M3/M4) | macOS development | MLX acceleration, OCRMac |
| `CloudGpuConfig` | NVIDIA GPU (CUDA) | Cloud/production | CUDA acceleration, EasyOCR |
| `CloudCpuConfig` | CPU-only | Universal | Lightweight, Tesseract OCR |

### Environment Variables

```bash
# General Configuration
export VERTECTOR_LOG_LEVEL=INFO
export VERTECTOR_CACHE_DIR=/path/to/cache

# VLM Configuration
export VERTECTOR_VLM_USE_MLX=true
export VERTECTOR_VLM_PRESET_MODEL=granite-mlx
export VERTECTOR_VLM_BATCH_SIZE=8

# Audio Configuration
export VERTECTOR_AUDIO_MODEL_SIZE=base
export VERTECTOR_AUDIO_BACKEND=mlx
export VERTECTOR_AUDIO_LANGUAGE=en

# OCR Configuration
export VERTECTOR_OCR_ENGINE=easyocr
export VERTECTOR_OCR_USE_GPU=true

# Vector Store Configuration
export VERTECTOR_VECTOR_STORE=chroma
export VERTECTOR_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
```

### Programmatic Configuration

```python
from vertector_data_ingestion.models.config import (
    ConverterConfig,
    VlmConfig,
    AudioConfig,
    OcrConfig,
    OcrEngine,
    WhisperModelSize,
)

# Create custom configuration
config = ConverterConfig(
    # VLM settings
    vlm=VlmConfig(
        use_mlx=True,
        preset_model="qwen25-3b",
        batch_size=16,
    ),

    # Audio settings
    audio=AudioConfig(
        model_size=WhisperModelSize.SMALL,
        language="es",
        beam_size=10,
    ),

    # OCR settings
    ocr=OcrConfig(
        engine=OcrEngine.EASYOCR,
        use_gpu=True,
        languages=["en", "es"],
    ),

    # Performance
    batch_processing_workers=8,
    enable_cache=True,
)
```

For complete configuration documentation, see [Configuration Guide](docs/configuration.md).

## Documentation

- [Installation Guide](docs/installation.md) - Detailed installation instructions
- [Quick Start](docs/quickstart.md) - Get started quickly
- [User Guide](docs/user-guide.md) - Comprehensive usage guide
- [MCP Server](docs/mcp-server.md) - Model Context Protocol integration
- [Configuration](docs/configuration.md) - Configuration options
- [API Reference](docs/api-reference.md) - API documentation
- [Examples](examples/) - Code examples

## Performance

### Hardware Recommendations

| Modality | Best Hardware | Recommended Model | Performance |
|----------|--------------|-------------------|-------------|
| Documents (VLM) | Apple M-series / NVIDIA GPU | Granite-Docling-258M | ~10-20x faster |
| Documents (Classic) | Any | N/A | Universal |
| Audio | Apple M-series / NVIDIA GPU | Whisper Base/Small | ~15x faster (MLX) |
| OCR | GPU-accelerated | EasyOCR | GPU-optimized |

### Optimization Tips

1. **Use MLX on Apple Silicon**: 10-20x faster than CPU for VLM/audio
2. **Enable caching**: Reduces redundant processing
3. **Batch processing**: Use `convert_batch()` for multiple files
4. **Choose appropriate models**: Smaller models (base, small) for faster processing
5. **Adjust batch sizes**: Increase for better GPU utilization

## Architecture

```
vertector-data-ingestion/
├── src/vertector_data_ingestion/
│   ├── core/              # Pipeline routing & hardware detection
│   ├── audio/             # Audio transcription (Whisper)
│   ├── ocr/               # OCR engines (EasyOCR, Tesseract, OCRMac)
│   ├── chunkers/          # RAG chunking strategies
│   ├── exporters/         # Export formats (Markdown, JSON, DocTags)
│   ├── models/            # Data models & configuration
│   ├── vector/            # Vector store adapters
│   ├── mcp/               # MCP server implementation
│   └── monitoring/        # Logging & metrics
├── tests/
│   ├── integration/       # Integration tests
│   ├── unit/              # Unit tests
│   └── fixtures/          # Test fixtures
├── examples/              # Usage examples
└── docs/                  # Documentation
```

## Testing

```bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=src/vertector_data_ingestion --cov-report=term

# Run integration tests
uv run pytest tests/integration/

# Run specific test
uv run pytest tests/integration/test_audio_transcription.py

# Run MCP server tests
uv run pytest tests/integration/test_mcp_server.py
```

## Known Limitations

### Image Processing
- Docling has known issues with standalone image files (PNG, JPG)
- Use VLM pipeline for better image understanding
- Embedded images in PDFs work correctly

### Audio
- MLX Whisper requires Apple Silicon (M1/M2/M3/M4)
- Standard Whisper requires significant memory for large models
- Word-level timestamps may not be available in all backends

## Troubleshooting

### Common Issues

**"MLX not available"**
```bash
# Install MLX Whisper for Apple Silicon
uv sync --extra mlx
```

**"CUDA out of memory"**
```python
# Reduce batch size
config.vlm.batch_size = 4
config.audio.model_size = WhisperModelSize.TINY
```

**"OCR not detecting text"**
```python
# Try different OCR engine
config.ocr.engine = OcrEngine.TESSERACT
# Or adjust confidence threshold
config.ocr.confidence_threshold = 0.3
```

For more troubleshooting help, see [Installation Guide - Troubleshooting](docs/installation.md#troubleshooting).

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Setup

```bash
# Clone repository
git clone https://github.com/vertector/vertector-data-ingestion.git
cd vertector-data-ingestion

# Install with development dependencies
uv sync --extra dev

# Install pre-commit hooks
pre-commit install

# Run tests
uv run pytest

# Run linting
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Run type checking
uv run mypy src/
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

**Enoch Tetteh**
Email: cutetetteh@gmail.com

## Acknowledgments

Built on top of:
- [IBM Docling](https://github.com/DS4SD/docling) - Document understanding
- [OpenAI Whisper](https://github.com/openai/whisper) - Audio transcription
- [MLX](https://github.com/ml-explore/mlx) - Apple Silicon acceleration
- [ChromaDB](https://www.trychroma.com/) - Vector database
