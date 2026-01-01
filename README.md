# Vertector Data Ingestion

Production-grade universal multimodal data ingestion pipeline for documents, images, and audio.

## Overview

Vertector is a comprehensive data ingestion system built on IBM's Docling framework, designed for RAG (Retrieval-Augmented Generation) applications. It provides unified processing for multiple data modalities with hardware acceleration and production-ready features.

## Features

### Multimodal Support
- **Documents**: PDF, DOCX, PPTX, XLSX, HTML with advanced table detection
- **Images**: Standalone image processing with VLM (Vision-Language Models)
- **Audio**: Speech-to-text transcription with Whisper (MLX/CUDA/CPU)

### Intelligent Processing
- **Dual Pipeline Architecture**: Auto-routing between Classic and VLM pipelines
- **Hardware Acceleration**: Auto-detect and optimize for MPS (Apple Silicon), CUDA (NVIDIA), or CPU
- **OCR Engine Selection**: EasyOCR, Tesseract, OCRMac (macOS native)
- **VLM Model Presets**: Granite, SmolDocling, Qwen2.5-VL, Pixtral, Gemma3, Phi4

### RAG-Ready Output
- **Hierarchical Chunking**: Token-aware chunking with metadata enrichment
- **Multiple Export Formats**: Markdown, JSON, DocTags
- **Vector Store Integration**: ChromaDB, Pinecone, Qdrant, OpenSearch
- **Embedding Support**: Qwen3-Embedding-4B (32K context window)

### Production Features
- **Unified Configuration**: Environment variables, config files, or programmatic
- **Caching & Retry Logic**: Configurable caching with intelligent retry
- **Batch Processing**: Multi-worker parallel processing
- **Monitoring**: Structured logging and metrics collection

## Installation

### Prerequisites
- Python 3.10+
- uv package manager

### Basic Installation

```bash
# Clone repository
git clone <repository-url>
cd vertector-data-ingestion

# Install dependencies
uv sync

# For development
uv sync --extra dev
```

### Optional Dependencies

```bash
# For audio transcription (choose one)
uv add openai-whisper              # Standard Whisper (CUDA/CPU)
uv add mlx-whisper                  # MLX Whisper (Apple Silicon)

# For specific OCR engines
uv add easyocr                      # EasyOCR
uv add pytesseract tesseract        # Tesseract
```

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

### Using Configuration Presets

```python
from vertector_data_ingestion import LocalMpsConfig, UniversalConverter

# Use optimized config for Apple Silicon
config = LocalMpsConfig()
converter = UniversalConverter(config)

# Process document
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
    backend=AudioBackend.MLX,  # or AUTO for auto-detection
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
doc = converter.convert_single("research_paper.pdf")

# Create chunks for RAG
chunker = HybridChunker()
chunks = chunker.chunk_document(doc.document)

# Store in vector database
vector_store = ChromaAdapter(collection_name="research_papers")
vector_store.add_chunks(chunks)
```

## Configuration

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
export VERTECTOR_AUDIO_BEAM_SIZE=5

# OCR Configuration
export VERTECTOR_OCR_ENGINE=easyocr
export VERTECTOR_OCR_USE_GPU=true

# Vector Store Configuration
export VERTECTOR_VECTOR_STORE=chroma
export VERTECTOR_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-4B
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
│   └── monitoring/        # Logging & metrics
├── tests/
│   ├── integration/       # Integration tests
│   ├── unit/              # Unit tests
│   └── fixtures/          # Test fixtures
└── examples/              # Usage examples
```

## Configuration Presets

### LocalMpsConfig (macOS/Apple Silicon)
- MLX acceleration for VLM and audio
- OCRMac for native macOS OCR
- Optimized batch sizes

### CloudGpuConfig (CUDA/Cloud)
- CUDA acceleration
- EasyOCR with GPU
- Larger batch sizes
- Production-ready defaults

### CloudCpuConfig (CPU-only)
- CPU-optimized settings
- Tesseract OCR
- Conservative batch sizes

## Advanced Features

### Custom VLM Models

```python
from vertector_data_ingestion.models.config import VlmConfig

vlm_config = VlmConfig(
    use_mlx=True,
    custom_model_repo_id="your-org/custom-vlm-model",
    custom_model_prompt="Custom prompt for your model",
    enable_picture_description=True,
)
```

### Batch Processing

```python
from pathlib import Path

# Process multiple documents
documents = [
    Path("doc1.pdf"),
    Path("doc2.docx"),
    Path("doc3.pptx"),
]

results = converter.convert_batch(documents)
```

### Pipeline Selection

```python
from vertector_data_ingestion import PipelineRouter, PipelineType

router = PipelineRouter(config)

# Force specific pipeline
doc = router.route_document(
    path="document.pdf",
    override_pipeline=PipelineType.VLM  # or PipelineType.CLASSIC
)
```

## Testing

```bash
# Run all tests
uv run pytest

# Run integration tests
uv run pytest tests/integration/

# Run specific test
uv run pytest tests/integration/test_audio_transcription.py

# With coverage
uv run pytest --cov=src/vertector_data_ingestion
```

## Performance

### Hardware Recommendations

| Modality | Best Hardware | Recommended Model |
|----------|--------------|-------------------|
| Documents (VLM) | Apple M-series / NVIDIA GPU | Granite-Docling-258M, Qwen2.5-VL-3B |
| Documents (Classic) | Any | N/A |
| Audio | Apple M-series / NVIDIA GPU | Whisper Base/Small |
| OCR | GPU-accelerated | EasyOCR |

### Optimization Tips

1. **Use MLX on Apple Silicon**: 10-20x faster than CPU for VLM/audio
2. **Enable caching**: Reduces redundant processing
3. **Batch processing**: Use `convert_batch()` for multiple files
4. **Choose appropriate models**: Smaller models (base, small) for faster processing
5. **Adjust batch sizes**: Increase for better GPU utilization

## Known Limitations

### Image Processing
- Docling has known issues with standalone image files (PNG, JPG)
- Use VLM pipeline for better image understanding
- Embedded images in PDFs work correctly

### Audio
- MLX Whisper requires Apple Silicon (M1/M2/M3)
- Standard Whisper requires significant memory for large models
- Word-level timestamps may not be available in all backends

## Troubleshooting

### Common Issues

**"MLX not available"**
```bash
# Install MLX Whisper for Apple Silicon
uv add mlx-whisper
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

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## License

[Your License Here]

## Contact

**Enoch Tetteh**
Email: cutetetteh@gmail.com
