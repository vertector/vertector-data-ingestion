# User Guide

Comprehensive guide to Vertector Data Ingestion features and capabilities.

## Table of Contents

- [Core Concepts](#core-concepts)
- [Document Processing](#document-processing)
- [Audio Transcription](#audio-transcription)
- [RAG Pipeline](#rag-pipeline)
- [Pipeline Selection](#pipeline-selection)
- [Hardware Acceleration](#hardware-acceleration)
- [Export Formats](#export-formats)
- [Vector Stores](#vector-stores)
- [Monitoring](#monitoring)
- [Known Limitations](#known-limitations)
- [Troubleshooting](#troubleshooting)

## Core Concepts

### Multimodal Processing

Vertector handles three primary data modalities:

1. **Documents**: PDF, DOCX, PPTX, XLSX, HTML
2. **Images**: Standalone PNG, JPG (via VLM pipeline)
3. **Audio**: MP3, WAV, M4A (via Whisper transcription)

### Dual Pipeline Architecture

Vertector uses two complementary pipelines:

- **Classic Pipeline**: Fast, rule-based document processing
- **VLM Pipeline**: AI-powered understanding using Vision-Language Models

The router automatically selects the best pipeline based on document characteristics.

### Hardware Detection

Automatic hardware detection and optimization:

- **Apple Silicon (MPS)**: MLX acceleration for VLM and audio
- **NVIDIA (CUDA)**: GPU acceleration for all modalities
- **CPU**: Optimized fallback for all systems

## Document Processing

### Supported Formats

| Format | Extension | Features |
|--------|-----------|----------|
| PDF | `.pdf` | Text extraction, OCR, table detection, images |
| Word | `.docx` | Text, tables, formatting |
| PowerPoint | `.pptx` | Slides, text, images |
| Excel | `.xlsx` | Sheets, tables, formulas |
| HTML | `.html` | Web content, structure |

### Basic Document Conversion

```python
from pathlib import Path
from vertector_data_ingestion import UniversalConverter

converter = UniversalConverter()

# Convert single document
doc = converter.convert_single(Path("document.pdf"))

# Access metadata
print(f"Pages: {doc.metadata.num_pages}")
print(f"Pipeline: {doc.metadata.pipeline_used}")
print(f"Time: {doc.metadata.processing_time:.2f}s")

# Get text content
text = doc.get_text()
print(text)
```

### Batch Processing

Process multiple documents efficiently:

```python
from pathlib import Path
from vertector_data_ingestion import UniversalConverter

converter = UniversalConverter()

# Process directory
documents = list(Path("documents/").glob("*.pdf"))
results = converter.convert_batch(documents)

for doc in results:
    print(f"{doc.metadata.source_path.name}: {doc.metadata.num_pages} pages")
```

### Table Extraction

Extract tables with structure preservation:

```python
from vertector_data_ingestion import UniversalConverter

converter = UniversalConverter()
doc = converter.convert_single("financial_report.pdf")

# Access tables
for table in doc.document.tables:
    print(f"Table on page {table.prov[0].page_no}")
    # Table data is preserved in structure
```

### OCR Configuration

Choose OCR engine based on your needs:

```python
from vertector_data_ingestion import ConverterConfig
from vertector_data_ingestion.models.config import OcrConfig, OcrEngine

config = ConverterConfig(
    ocr=OcrConfig(
        engine=OcrEngine.EASYOCR,  # or TESSERACT, OCRMAC
        use_gpu=True,
        languages=["en", "es"],
        confidence_threshold=0.5,
    )
)

converter = UniversalConverter(config)
```

## Audio Transcription

### Basic Transcription

```python
from pathlib import Path
from vertector_data_ingestion import (
    create_audio_transcriber,
    AudioConfig,
    WhisperModelSize,
    AudioBackend,
)

# Configure transcriber
config = AudioConfig(
    model_size=WhisperModelSize.BASE,
    backend=AudioBackend.AUTO,  # Auto-detect best backend
    language="en",
    word_timestamps=True,
)

transcriber = create_audio_transcriber(config)

# Transcribe
result = transcriber.transcribe(Path("meeting.wav"))

print(result.text)
print(f"Language: {result.language}")
print(f"Duration: {result.duration:.2f}s")
```

### Timestamped Segments

Access word and segment-level timestamps:

```python
# Get segments with timestamps
for segment in result.segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s]")
    print(f"  {segment.text}")
```

### Model Selection

Choose model size based on accuracy vs. speed tradeoff:

| Model | Speed | Accuracy | Use Case |
|-------|-------|----------|----------|
| `TINY` | Fastest | Good | Real-time, drafts |
| `BASE` | Fast | Very Good | General purpose |
| `SMALL` | Medium | Excellent | High accuracy needed |
| `MEDIUM` | Slow | Excellent | Professional transcription |
| `LARGE` | Slowest | Best | Maximum accuracy |
| `TURBO` | Fast | Excellent | Large-scale processing |

```python
from vertector_data_ingestion import AudioConfig, WhisperModelSize

# Fast transcription
fast_config = AudioConfig(model_size=WhisperModelSize.TINY)

# High accuracy
accurate_config = AudioConfig(model_size=WhisperModelSize.LARGE)
```

### Advanced Options

Fine-tune transcription quality:

```python
from vertector_data_ingestion import AudioConfig

config = AudioConfig(
    model_size=WhisperModelSize.BASE,
    language="en",  # Force language (faster than auto-detect)
    word_timestamps=True,  # Enable word-level timestamps
    initial_prompt="Technical discussion about quantum computing",  # Context hint
    beam_size=10,  # Higher = more accurate, slower
    temperature=0.0,  # Deterministic output
    vad_filter=True,  # Filter silence
    condition_on_previous_text=True,  # Use context
)
```

## RAG Pipeline

### Document Chunking

Create RAG-optimized chunks:

```python
from vertector_data_ingestion import (
    UniversalConverter,
    HybridChunker,
)

# Convert document
converter = UniversalConverter()
doc = converter.convert_single("research_paper.pdf")

# Create chunks
chunker = HybridChunker()
chunking_result = chunker.chunk_document(doc.document)

print(f"Chunks: {chunking_result.total_chunks}")
print(f"Total tokens: {chunking_result.total_tokens}")
print(f"Avg size: {chunking_result.avg_chunk_size:.1f} tokens")

# Access individual chunks
for chunk in chunking_result.chunks:
    print(f"Page {chunk.metadata.page_no}: {chunk.text[:100]}...")
```

### Chunking Configuration

Customize chunking strategy:

```python
from vertector_data_ingestion import ConverterConfig
from vertector_data_ingestion.models.config import ChunkingConfig

config = ConverterConfig(
    chunking=ChunkingConfig(
        max_tokens=512,  # Maximum chunk size
        overlap_tokens=50,  # Overlap between chunks
        min_chunk_size=100,  # Minimum viable chunk
        respect_boundaries=True,  # Respect section boundaries
    )
)

converter = UniversalConverter(config)
```

### Metadata Enrichment

Chunks include rich metadata:

```python
for chunk in chunking_result.chunks:
    print(f"Chunk {chunk.chunk_id}")
    print(f"  Page: {chunk.metadata.page_no}")
    print(f"  Tokens: {chunk.metadata.token_count}")
    print(f"  Heading: {chunk.metadata.get('heading', 'N/A')}")
    print(f"  Section: {chunk.metadata.get('section', 'N/A')}")
```

## Pipeline Selection

### Automatic Selection

Router automatically chooses the best pipeline:

```python
from vertector_data_ingestion import UniversalConverter

converter = UniversalConverter()

# Auto-routing based on document characteristics
doc = converter.convert_single("document.pdf")
print(f"Used: {doc.metadata.pipeline_used}")
```

### Manual Override

Force specific pipeline:

```python
from vertector_data_ingestion import PipelineRouter, PipelineType

router = PipelineRouter(config)

# Force Classic pipeline (fast)
doc = router.route_document(
    path="document.pdf",
    override_pipeline=PipelineType.CLASSIC
)

# Force VLM pipeline (AI-powered)
doc = router.route_document(
    path="document.pdf",
    override_pipeline=PipelineType.VLM
)
```

### Pipeline Comparison

| Feature | Classic | VLM |
|---------|---------|-----|
| Speed | Faster | Slower |
| Accuracy | Good | Excellent |
| Images | Basic | Advanced understanding |
| Tables | Rule-based | AI-powered |
| Complex layouts | Limited | Excellent |
| Hardware | CPU-friendly | GPU-preferred |

## Hardware Acceleration

### Apple Silicon (MLX)

Optimize for M1/M2/M3:

```python
from vertector_data_ingestion import LocalMpsConfig, UniversalConverter

config = LocalMpsConfig()
print(f"MLX enabled: {config.vlm.use_mlx}")
print(f"Audio backend: {config.audio.backend}")

converter = UniversalConverter(config)
```

### NVIDIA GPU (CUDA)

Optimize for GPU:

```python
from vertector_data_ingestion import CloudGpuConfig, UniversalConverter

config = CloudGpuConfig()
print(f"OCR GPU: {config.ocr.use_gpu}")
print(f"VLM batch size: {config.vlm.batch_size}")

converter = UniversalConverter(config)
```

### CPU Optimization

Optimize for CPU:

```python
from vertector_data_ingestion import CloudCpuConfig, UniversalConverter

config = CloudCpuConfig()
print(f"OCR engine: {config.ocr.engine}")
print(f"Workers: {config.batch_processing_workers}")

converter = UniversalConverter(config)
```

## Export Formats

### Markdown

Export to Markdown:

```python
from vertector_data_ingestion import UniversalConverter, ExportFormat

converter = UniversalConverter()
doc = converter.convert_single("document.pdf")

# Export
markdown = converter.export(doc, ExportFormat.MARKDOWN)

# Save to file
converter.convert_and_export(
    source="document.pdf",
    output_path="output.md",
    format=ExportFormat.MARKDOWN
)
```

### JSON

Export to structured JSON:

```python
json_output = converter.export(doc, ExportFormat.JSON)

# JSON includes full document structure
import json
data = json.loads(json_output)
print(data.keys())  # metadata, pages, tables, etc.
```

### DocTags

Export to DocTags format (for advanced processing):

```python
doctags = converter.export(doc, ExportFormat.DOCTAGS)
```

## Vector Stores

### ChromaDB

```python
from pathlib import Path
from vertector_data_ingestion import (
    UniversalConverter,
    HybridChunker,
    ChromaAdapter,
)

# Convert and chunk
converter = UniversalConverter()
doc = converter.convert_single("document.pdf")
chunks = HybridChunker().chunk_document(doc.document)

# Store in ChromaDB
chroma = ChromaAdapter(
    collection_name="my_docs",
    persist_directory=Path("./chroma_db")
)

chroma.add_chunks(chunks.chunks)

# Search
results = chroma.search("quantum computing", top_k=5)
for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text'][:100]}...")
```

### Other Vector Stores

Vertector provides adapters for:

- **Pinecone**: Cloud-native vector database
- **Qdrant**: High-performance vector search
- **OpenSearch**: Elasticsearch-based vector search

## Monitoring

### Logging

Configure logging level:

```python
from vertector_data_ingestion import setup_logging

# Set log level
setup_logging(log_level="INFO")  # DEBUG, INFO, WARNING, ERROR

# Logs include:
# - Pipeline selection decisions
# - Processing progress
# - Performance metrics
# - Error details
```

### Metrics Collection

Track processing metrics:

```python
from vertector_data_ingestion import MetricsCollector

metrics = MetricsCollector()

# Metrics are automatically collected:
# - Processing times
# - Token counts
# - Chunk statistics
# - Error rates
```

## Known Limitations

### Image Processing

- **Standalone Images**: Docling has issues with PNG/JPG files
  - **Workaround**: Use VLM pipeline for better image understanding
  - **Note**: Embedded images in PDFs work correctly

### Audio

- **MLX Requirement**: MLX Whisper requires Apple Silicon
- **Memory Usage**: Large models require significant memory
- **Word Timestamps**: May not be available in all backends

### Document Processing

- **Complex Layouts**: Very complex layouts may require VLM pipeline
- **Handwriting**: OCR accuracy varies with handwriting quality
- **Non-Latin Scripts**: Ensure OCR engine supports required languages

## Troubleshooting

### Slow Processing

**Problem**: Document processing is slow

**Solutions**:
1. Use hardware acceleration (MPS/CUDA)
2. Reduce VLM batch size if running out of memory
3. Use Classic pipeline for simple documents
4. Enable caching to avoid reprocessing

```python
config = LocalMpsConfig()
config.enable_cache = True
config.vlm.batch_size = 4  # Reduce if needed
```

### OCR Not Detecting Text

**Problem**: OCR fails to detect text in images

**Solutions**:
1. Try different OCR engine
2. Adjust confidence threshold
3. Verify image quality
4. Check language settings

```python
from vertector_data_ingestion.models.config import OcrConfig, OcrEngine

config = OcrConfig(
    engine=OcrEngine.EASYOCR,  # Try different engine
    confidence_threshold=0.3,  # Lower threshold
    languages=["en"],  # Specify languages
)
```

### Memory Errors

**Problem**: Out of memory errors

**Solutions**:
1. Use smaller models
2. Reduce batch sizes
3. Process documents individually
4. Clear cache

```python
from vertector_data_ingestion import CloudCpuConfig, WhisperModelSize

config = CloudCpuConfig()
config.vlm.batch_size = 2
config.audio.model_size = WhisperModelSize.TINY
```

### Audio Transcription Errors

**Problem**: Transcription quality is poor

**Solutions**:
1. Use larger model
2. Specify language explicitly
3. Provide context via initial_prompt
4. Increase beam_size

```python
from vertector_data_ingestion import AudioConfig, WhisperModelSize

config = AudioConfig(
    model_size=WhisperModelSize.SMALL,  # Larger model
    language="en",  # Explicit language
    initial_prompt="Discussion about machine learning",  # Context
    beam_size=10,  # Higher accuracy
)
```

### Import Errors

**Problem**: Cannot import modules

**Solutions**:
1. Verify installation: `uv sync`
2. Check Python version: `python --version` (need 3.10+)
3. Activate virtual environment
4. Install optional dependencies for specific features

```bash
# Reinstall
uv sync

# Check Python version
python --version

# Install optional dependencies
uv add mlx-whisper  # For Apple Silicon audio
uv add easyocr     # For EasyOCR
```

## Performance Tips

### 1. Use Appropriate Hardware

- **Apple Silicon**: Enable MLX for 10-20x speedup
- **NVIDIA GPU**: Enable GPU acceleration
- **CPU**: Use optimized models and batch sizes

### 2. Enable Caching

```python
config = ConverterConfig()
config.enable_cache = True
config.cache_dir = Path("./cache")
```

### 3. Batch Processing

Process multiple documents together:

```python
# More efficient than individual processing
results = converter.convert_batch(documents)
```

### 4. Choose Right Models

- **Documents**: Use Classic pipeline for simple docs
- **Audio**: Use BASE model for general purpose
- **VLM**: Use smaller models (Granite, SmolDocling) when possible

### 5. Optimize Chunking

```python
# Larger chunks = fewer chunks = faster embedding
chunking_config = ChunkingConfig(
    max_tokens=1024,  # Increase for fewer chunks
    overlap_tokens=100,  # Balance context vs. speed
)
```

## Next Steps

- Review [Configuration Guide](configuration.md) for detailed settings
- Check [API Reference](api-reference.md) for complete API documentation
- Explore [examples/](../examples/) for more code samples

## Getting Help

Contact: cutetetteh@gmail.com
