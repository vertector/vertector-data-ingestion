# API Reference

Complete API documentation for Vertector Data Ingestion.

## Table of Contents

- [Core Classes](#core-classes)
- [Configuration](#configuration)
- [Audio](#audio)
- [Chunking](#chunking)
- [Vector Stores](#vector-stores)
- [Models](#models)
- [Utilities](#utilities)

## Core Classes

### UniversalConverter

Main entry point for document conversion.

```python
from vertector_data_ingestion import UniversalConverter, ConverterConfig
```

#### Constructor

```python
UniversalConverter(config: Optional[ConverterConfig] = None)
```

**Parameters**:
- `config`: Configuration object. If None, uses default config.

**Example**:
```python
converter = UniversalConverter()
# or
converter = UniversalConverter(LocalMpsConfig())
```

#### Methods

##### convert_single

Convert a single document.

```python
convert_single(
    source: Union[str, Path],
    **kwargs
) -> DoclingDocumentWrapper
```

**Parameters**:
- `source`: Path to document file
- `**kwargs`: Additional conversion options

**Returns**: `DoclingDocumentWrapper` with conversion result

**Example**:
```python
doc = converter.convert_single("document.pdf")
```

##### convert_batch

Convert multiple documents in parallel.

```python
convert_batch(
    sources: List[Union[str, Path]],
    **kwargs
) -> List[DoclingDocumentWrapper]
```

**Parameters**:
- `sources`: List of document paths
- `**kwargs`: Additional conversion options

**Returns**: List of `DoclingDocumentWrapper` objects

**Example**:
```python
docs = converter.convert_batch(["doc1.pdf", "doc2.docx"])
```

##### export

Export document to specified format.

```python
export(
    doc: DoclingDocumentWrapper,
    format: ExportFormat
) -> str
```

**Parameters**:
- `doc`: Document wrapper to export
- `format`: Export format (MARKDOWN, JSON, DOCTAGS)

**Returns**: Exported content as string

**Example**:
```python
markdown = converter.export(doc, ExportFormat.MARKDOWN)
```

##### convert_and_export

Convert and export in one step.

```python
convert_and_export(
    source: Union[str, Path],
    output_path: Path,
    format: ExportFormat = ExportFormat.MARKDOWN,
) -> None
```

**Parameters**:
- `source`: Input document path
- `output_path`: Output file path
- `format`: Export format

**Example**:
```python
converter.convert_and_export(
    source="input.pdf",
    output_path="output.md",
    format=ExportFormat.MARKDOWN
)
```

### PipelineRouter

Routes documents to appropriate processing pipeline.

```python
from vertector_data_ingestion import PipelineRouter
```

#### Constructor

```python
PipelineRouter(config: ConverterConfig)
```

#### Methods

##### route_document

Process document with selected pipeline.

```python
route_document(
    path: Union[str, Path],
    override_pipeline: Optional[PipelineType] = None
) -> DoclingDocumentWrapper
```

**Parameters**:
- `path`: Document path
- `override_pipeline`: Force specific pipeline (CLASSIC or VLM)

**Returns**: Processed document

**Example**:
```python
router = PipelineRouter(config)
doc = router.route_document("doc.pdf", override_pipeline=PipelineType.VLM)
```

### HardwareDetector

Detects available hardware acceleration.

```python
from vertector_data_ingestion import HardwareDetector
```

#### Methods

##### detect (static)

Detect hardware configuration.

```python
HardwareDetector.detect() -> HardwareConfig
```

**Returns**: Hardware configuration object

**Example**:
```python
hw_config = HardwareDetector.detect()
print(hw_config.device_type)  # mps, cuda, or cpu
```

##### get_device_info (static)

Get detailed hardware information.

```python
HardwareDetector.get_device_info() -> Dict[str, Any]
```

**Returns**: Dictionary with hardware details

**Example**:
```python
info = HardwareDetector.get_device_info()
for key, value in info.items():
    print(f"{key}: {value}")
```

## Configuration

### ConverterConfig

Main configuration class.

```python
from vertector_data_ingestion import ConverterConfig
```

#### Constructor

```python
ConverterConfig(
    log_level: str = "INFO",
    enable_cache: bool = True,
    cache_dir: Path = Path("./cache"),
    cache_ttl: int = 86400,
    batch_processing_workers: int = 4,
    max_retries: int = 3,
    retry_delay: float = 1.0,
    vlm: Optional[VlmConfig] = None,
    audio: Optional[AudioConfig] = None,
    ocr: Optional[OcrConfig] = None,
    chunking: Optional[ChunkingConfig] = None,
)
```

### Configuration Presets

#### LocalMpsConfig

```python
from vertector_data_ingestion import LocalMpsConfig

config = LocalMpsConfig()
```

Preset for Apple Silicon development.

#### CloudGpuConfig

```python
from vertector_data_ingestion import CloudGpuConfig

config = CloudGpuConfig()
```

Preset for GPU cloud deployment.

#### CloudCpuConfig

```python
from vertector_data_ingestion import CloudCpuConfig

config = CloudCpuConfig()
```

Preset for CPU-only deployment.

### VlmConfig

Vision-Language Model configuration.

```python
from vertector_data_ingestion.models.config import VlmConfig

vlm = VlmConfig(
    use_mlx: bool = False,
    preset_model: Optional[str] = None,
    custom_model_repo_id: Optional[str] = None,
    custom_model_prompt: Optional[str] = None,
    batch_size: int = 8,
    enable_picture_description: bool = True,
    enable_table_detection: bool = True,
)
```

### AudioConfig

Audio transcription configuration.

```python
from vertector_data_ingestion import AudioConfig, WhisperModelSize, AudioBackend

audio = AudioConfig(
    model_size: WhisperModelSize = WhisperModelSize.BASE,
    backend: AudioBackend = AudioBackend.AUTO,
    language: Optional[str] = None,
    word_timestamps: bool = True,
    initial_prompt: Optional[str] = None,
    beam_size: int = 5,
    temperature: float = 0.0,
    vad_filter: bool = False,
    condition_on_previous_text: bool = True,
)
```

### OcrConfig

OCR configuration.

```python
from vertector_data_ingestion.models.config import OcrConfig, OcrEngine

ocr = OcrConfig(
    engine: OcrEngine = OcrEngine.EASYOCR,
    use_gpu: bool = False,
    languages: List[str] = ["en"],
    confidence_threshold: float = 0.5,
    batch_size: int = 4,
)
```

### ChunkingConfig

Chunking strategy configuration.

```python
from vertector_data_ingestion.models.config import ChunkingConfig

chunking = ChunkingConfig(
    max_tokens: int = 512,
    min_chunk_size: int = 100,
    overlap_tokens: int = 50,
    respect_boundaries: bool = True,
    respect_sentences: bool = True,
    include_metadata: bool = True,
)
```

## Audio

### create_audio_transcriber

Factory function for creating transcribers.

```python
from vertector_data_ingestion import create_audio_transcriber, AudioConfig

transcriber = create_audio_transcriber(config: AudioConfig)
```

**Parameters**:
- `config`: Audio configuration

**Returns**: Configured `AudioTranscriber` instance

**Example**:
```python
config = AudioConfig(model_size=WhisperModelSize.BASE)
transcriber = create_audio_transcriber(config)
```

### AudioTranscriber

Base class for audio transcription.

#### Methods

##### transcribe

Transcribe audio file.

```python
transcribe(
    audio_path: Path,
    language: Optional[str] = None
) -> TranscriptionResult
```

**Parameters**:
- `audio_path`: Path to audio file
- `language`: Language code (overrides config if provided)

**Returns**: `TranscriptionResult` object

**Example**:
```python
result = transcriber.transcribe(Path("audio.wav"))
print(result.text)
```

##### is_available

Check if transcriber is available.

```python
is_available() -> bool
```

**Returns**: True if transcriber can be used

### TranscriptionResult

Result of audio transcription.

#### Attributes

```python
class TranscriptionResult:
    text: str                        # Full transcription
    language: str                    # Detected/specified language
    segments: List[TranscriptionSegment]  # Timestamped segments
    duration: float                  # Processing duration
    model_name: str                  # Model used
```

**Example**:
```python
result = transcriber.transcribe(Path("audio.wav"))
print(f"Text: {result.text}")
print(f"Language: {result.language}")
print(f"Duration: {result.duration:.2f}s")
```

### TranscriptionSegment

Timestamped segment of transcription.

#### Attributes

```python
class TranscriptionSegment:
    start: float  # Start time in seconds
    end: float    # End time in seconds
    text: str     # Segment text
```

**Example**:
```python
for segment in result.segments:
    print(f"[{segment.start:.1f}s - {segment.end:.1f}s]: {segment.text}")
```

## Chunking

### HybridChunker

Creates RAG-optimized document chunks.

```python
from vertector_data_ingestion import HybridChunker
```

#### Constructor

```python
HybridChunker(config: Optional[ChunkingConfig] = None)
```

#### Methods

##### chunk_document

Create chunks from document.

```python
chunk_document(
    document: DoclingDocument
) -> ChunkingResult
```

**Parameters**:
- `document`: Docling document to chunk

**Returns**: `ChunkingResult` with chunks and statistics

**Example**:
```python
chunker = HybridChunker()
result = chunker.chunk_document(doc.document)
print(f"Created {result.total_chunks} chunks")
```

### ChunkingResult

Result of document chunking.

#### Attributes

```python
class ChunkingResult:
    chunks: List[DocumentChunk]  # List of chunks
    total_chunks: int             # Total number of chunks
    total_tokens: int             # Total tokens across all chunks
    avg_chunk_size: float         # Average tokens per chunk
```

### DocumentChunk

Individual document chunk.

#### Attributes

```python
class DocumentChunk:
    chunk_id: str                # Unique chunk identifier
    text: str                    # Chunk text content
    metadata: Dict[str, Any]     # Chunk metadata
```

**Metadata includes**:
- `page_no`: Page number
- `chunk_no`: Chunk number within page
- `token_count`: Number of tokens
- `heading`: Section heading (if available)
- `section`: Section name (if available)

**Example**:
```python
for chunk in result.chunks:
    print(f"Chunk {chunk.chunk_id}")
    print(f"  Page: {chunk.metadata['page_no']}")
    print(f"  Tokens: {chunk.metadata['token_count']}")
    print(f"  Text: {chunk.text[:100]}...")
```

## Vector Stores

### ChromaAdapter

ChromaDB vector store adapter.

```python
from vertector_data_ingestion import ChromaAdapter
```

#### Constructor

```python
ChromaAdapter(
    collection_name: str,
    persist_directory: Optional[Path] = None,
    embedding_model: Optional[str] = None,
)
```

**Parameters**:
- `collection_name`: Name of ChromaDB collection
- `persist_directory`: Directory for persistent storage
- `embedding_model`: HuggingFace model for embeddings

**Example**:
```python
chroma = ChromaAdapter(
    collection_name="my_docs",
    persist_directory=Path("./chroma_db")
)
```

#### Methods

##### add_chunks

Add document chunks to vector store.

```python
add_chunks(
    chunks: List[DocumentChunk],
    batch_size: int = 100
) -> None
```

**Parameters**:
- `chunks`: List of document chunks
- `batch_size`: Batch size for insertion

**Example**:
```python
chroma.add_chunks(chunking_result.chunks)
```

##### search

Search for similar chunks.

```python
search(
    query: str,
    top_k: int = 5,
    filter: Optional[Dict] = None
) -> List[Dict[str, Any]]
```

**Parameters**:
- `query`: Search query
- `top_k`: Number of results to return
- `filter`: Metadata filter

**Returns**: List of search results with text, metadata, and scores

**Example**:
```python
results = chroma.search("quantum computing", top_k=5)
for result in results:
    print(f"Score: {result['score']}")
    print(f"Text: {result['text']}")
```

##### delete_collection

Delete the collection.

```python
delete_collection() -> None
```

**Example**:
```python
chroma.delete_collection()
```

## Models

### DoclingDocumentWrapper

Wrapper for converted documents.

#### Attributes

```python
class DoclingDocumentWrapper:
    document: DoclingDocument    # Original Docling document
    metadata: DocumentMetadata   # Conversion metadata
```

#### Methods

##### get_text

Get document text.

```python
get_text() -> str
```

**Returns**: Full document text

##### export_to_markdown

Export to Markdown.

```python
export_to_markdown() -> str
```

**Returns**: Markdown-formatted document

##### export_to_json

Export to JSON.

```python
export_to_json() -> str
```

**Returns**: JSON-formatted document

### DocumentMetadata

Metadata about converted document.

#### Attributes

```python
class DocumentMetadata:
    source_path: Path        # Original file path
    num_pages: int           # Number of pages
    pipeline_used: str       # Pipeline used (classic/vlm)
    processing_time: float   # Processing time in seconds
    format: str              # Document format (pdf, docx, etc.)
```

## Utilities

### setup_logging

Configure logging.

```python
from vertector_data_ingestion import setup_logging

setup_logging(
    log_level: str = "INFO",
    log_file: Optional[Path] = None
)
```

**Parameters**:
- `log_level`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `log_file`: Optional log file path

**Example**:
```python
setup_logging(log_level="DEBUG", log_file=Path("app.log"))
```

### MetricsCollector

Collect processing metrics.

```python
from vertector_data_ingestion import MetricsCollector

metrics = MetricsCollector()
```

#### Methods

##### record_metric

Record a metric.

```python
record_metric(
    name: str,
    value: Union[int, float],
    tags: Optional[Dict[str, str]] = None
) -> None
```

##### get_metrics

Get recorded metrics.

```python
get_metrics() -> Dict[str, Any]
```

**Returns**: Dictionary of metrics

## Enums

### ExportFormat

Available export formats.

```python
from vertector_data_ingestion import ExportFormat

ExportFormat.MARKDOWN  # Markdown format
ExportFormat.JSON      # JSON format
ExportFormat.DOCTAGS   # DocTags format
```

### PipelineType

Processing pipeline types.

```python
from vertector_data_ingestion import PipelineType

PipelineType.CLASSIC  # Fast, rule-based pipeline
PipelineType.VLM      # AI-powered Vision-Language Model pipeline
```

### WhisperModelSize

Whisper model sizes.

```python
from vertector_data_ingestion import WhisperModelSize

WhisperModelSize.TINY
WhisperModelSize.BASE
WhisperModelSize.SMALL
WhisperModelSize.MEDIUM
WhisperModelSize.LARGE
WhisperModelSize.TURBO
```

### AudioBackend

Audio processing backends.

```python
from vertector_data_ingestion import AudioBackend

AudioBackend.AUTO      # Auto-detect best backend
AudioBackend.MLX       # MLX (Apple Silicon)
AudioBackend.STANDARD  # Standard PyTorch (CUDA/CPU)
```

### OcrEngine

OCR engine options.

```python
from vertector_data_ingestion.models.config import OcrEngine

OcrEngine.EASYOCR     # EasyOCR engine
OcrEngine.TESSERACT   # Tesseract OCR
OcrEngine.OCRMAC      # macOS native OCR
```

## Complete Example

```python
from pathlib import Path
from vertector_data_ingestion import (
    UniversalConverter,
    LocalMpsConfig,
    HybridChunker,
    ChromaAdapter,
    ExportFormat,
    create_audio_transcriber,
    AudioConfig,
    WhisperModelSize,
    setup_logging,
)

# Setup
setup_logging(log_level="INFO")

# Document processing
config = LocalMpsConfig()
converter = UniversalConverter(config)

doc = converter.convert_single("document.pdf")
markdown = converter.export(doc, ExportFormat.MARKDOWN)

# Audio transcription
audio_config = AudioConfig(model_size=WhisperModelSize.BASE)
transcriber = create_audio_transcriber(audio_config)
result = transcriber.transcribe(Path("audio.wav"))

# RAG pipeline
chunker = HybridChunker()
chunks = chunker.chunk_document(doc.document)

chroma = ChromaAdapter(collection_name="docs")
chroma.add_chunks(chunks.chunks)

results = chroma.search("quantum computing", top_k=5)
```

## See Also

- [QuickStart Guide](quickstart.md)
- [User Guide](user-guide.md)
- [Configuration Guide](configuration.md)
