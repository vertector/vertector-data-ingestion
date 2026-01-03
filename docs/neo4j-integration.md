# Neo4j SimpleKGPipeline Integration

Integration design for Vertector Data Ingestion with Neo4j's SimpleKGPipeline for end-to-end multimodal knowledge graph construction.

## Overview

This integration combines Vertector's multimodal data ingestion capabilities with Neo4j's knowledge graph construction, creating a powerful pipeline that can:

1. **Ingest multimodal content** (documents, images, audio) using Vertector
2. **Extract structured knowledge** using Neo4j's SimpleKGPipeline
3. **Build knowledge graphs** with rich metadata and relationships

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  Multimodal Knowledge Graph Pipeline                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────────────┐        ┌──────────────────────┐         │
│  │ Vertector Ingestion  │───────▶│ Neo4j SimpleKGPipeline│         │
│  │  - Documents         │        │  - Entity Extraction  │         │
│  │  - Images (VLM)      │        │  - Relation Extraction│         │
│  │  - Audio (Whisper)   │        │  - Schema Validation  │         │
│  │  - Tables            │        │  - Entity Resolution  │         │
│  └──────────────────────┘        └──────────────────────┘         │
│           │                                  │                      │
│           │                                  │                      │
│           ▼                                  ▼                      │
│  ┌──────────────────────┐        ┌──────────────────────┐         │
│  │ DocumentChunk[]      │───────▶│   Text Chunks        │         │
│  │  + metadata          │        │   (List[str])        │         │
│  │  + timestamps        │        │                       │         │
│  │  + bounding boxes    │        │                       │         │
│  └──────────────────────┘        └──────────────────────┘         │
│                                              │                      │
│                                              ▼                      │
│                                  ┌──────────────────────┐         │
│                                  │   Neo4j Graph DB     │         │
│                                  │   + Rich Metadata    │         │
│                                  └──────────────────────┘         │
└─────────────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. Custom Data Loader (Primary Integration)

Replace Neo4j's default `PdfLoader` with Vertector's `UniversalConverter`:

**Benefits:**
- Multiple formats: PDF, DOCX, PPTX, XLSX, HTML, Audio (MP3, WAV), Images
- Vision-Language Model pipeline for images
- Audio transcription with Whisper (MLX/CUDA/CPU)
- Advanced table extraction
- Rich metadata preservation

**Implementation:**
```python
from neo4j_graphrag.experimental.components.pdf_loader import DataLoader
from vertector_data_ingestion import UniversalConverter, ExportFormat, LocalMpsConfig
from pathlib import Path

class VertectorDataLoader(DataLoader):
    """Custom data loader using Vertector for multimodal ingestion."""

    def __init__(self, config=None):
        """
        Initialize loader with Vertector configuration.

        Args:
            config: LocalMpsConfig, CloudGpuConfig, CloudCpuConfig, or ConverterConfig
        """
        self.converter = UniversalConverter(config or LocalMpsConfig())
        self.last_metadata = {}

    async def load(self, path: str) -> str:
        """
        Load document using Vertector's unified API.

        Args:
            path: Path to document (PDF, DOCX, PPTX, XLSX, audio, etc.)

        Returns:
            Markdown text representation
        """
        path = Path(path)

        # Use unified convert() method - handles single files and batches
        doc_wrapper = self.converter.convert(path)

        # Export to markdown for text processing
        text = self.converter.export(doc_wrapper, ExportFormat.MARKDOWN)

        # Preserve metadata for graph enrichment
        self.last_metadata = {
            'source_path': str(path),
            'filename': path.name,
            'num_pages': doc_wrapper.metadata.num_pages,
            'pipeline_type': doc_wrapper.metadata.pipeline_type,
            'processing_time': doc_wrapper.metadata.processing_time,
            'file_size_bytes': path.stat().st_size,
        }

        return text
```

### 2. Audio File Loader

Handle audio files with transcription:

**Implementation:**
```python
from neo4j_graphrag.experimental.components.pdf_loader import DataLoader
from vertector_data_ingestion import create_audio_transcriber, AudioConfig, WhisperModelSize
from pathlib import Path

class VertectorAudioLoader(DataLoader):
    """Audio loader using Vertector's Whisper transcription."""

    def __init__(self, config=None):
        """
        Initialize audio transcriber.

        Args:
            config: AudioConfig or None for defaults
        """
        if config is None:
            config = AudioConfig(
                model_size=WhisperModelSize.BASE,
                word_timestamps=True,
            )
        self.transcriber = create_audio_transcriber(config)
        self.last_metadata = {}

    async def load(self, path: str) -> str:
        """
        Load and transcribe audio file.

        Args:
            path: Path to audio file (.wav, .mp3, .m4a, .flac)

        Returns:
            Transcribed text with timestamps
        """
        path = Path(path)

        # Transcribe audio
        result = self.transcriber.transcribe(path)

        # Format as markdown with timestamps
        markdown_parts = [f"# Audio Transcription: {path.name}\n"]
        markdown_parts.append(f"**Duration:** {result.duration:.2f}s\n")
        markdown_parts.append(f"**Language:** {result.language}\n\n")

        for i, segment in enumerate(result.segments, 1):
            timestamp = f"[{segment.start:.1f}s - {segment.end:.1f}s]"
            markdown_parts.append(f"**{i}.** {timestamp} {segment.text}\n")

        text = "\n".join(markdown_parts)

        # Preserve metadata
        self.last_metadata = {
            'source_path': str(path),
            'filename': path.name,
            'duration': result.duration,
            'language': result.language,
            'segments': len(result.segments),
            'model': result.model_name,
            'modality': 'audio',
        }

        return text
```

### 3. Enhanced Text Splitter with Metadata

Replace default splitter with Vertector's `HybridChunker`:

**Benefits:**
- Token-aware chunking (respects LLM token limits)
- Hierarchical structure preservation
- Section-aware splitting
- Rich chunk metadata (page numbers, bounding boxes, hierarchy)

**Implementation:**
```python
from neo4j_graphrag.experimental.components.text_splitters.text_splitter import TextSplitter
from vertector_data_ingestion import HybridChunker
from vertector_data_ingestion.models.config import ChunkingConfig
from vertector_data_ingestion.models.chunk import DocumentChunk
from typing import List

class VertectorTextSplitter(TextSplitter):
    """Text splitter using Vertector's HybridChunker."""

    def __init__(
        self,
        chunk_size: int = 512,
        tokenizer: str = "Qwen/Qwen3-Embedding-0.6B",
        include_metadata: bool = True
    ):
        """
        Initialize text splitter.

        Args:
            chunk_size: Maximum tokens per chunk
            tokenizer: HuggingFace tokenizer model
            include_metadata: Include metadata in chunk text
        """
        config = ChunkingConfig(
            tokenizer=tokenizer,
            max_tokens=chunk_size,
        )
        self.chunker = HybridChunker(config=config)
        self.include_metadata = include_metadata
        self.last_chunks = []  # Store DocumentChunk objects

    async def split(self, text: str) -> List[str]:
        """
        Split text using hybrid chunking strategy.

        Args:
            text: Input text (markdown, plain text, etc.)

        Returns:
            List of text chunks (Neo4j format: List[str])
        """
        # Use HybridChunker's text chunking capability
        chunking_result = self.chunker.chunk_text(text)

        # Store DocumentChunk objects for metadata access
        self.last_chunks = chunking_result.chunks

        # Convert to Neo4j text chunks (List[str])
        text_chunks = []
        for chunk in chunking_result.chunks:
            if self.include_metadata and chunk.page_no:
                # Optionally enrich with inline metadata
                chunk_text = f"{chunk.text}\n\n[Page: {chunk.page_no}]"
            else:
                chunk_text = chunk.text

            text_chunks.append(chunk_text)

        return text_chunks

    def get_chunk_metadata(self, chunk_index: int) -> dict:
        """
        Get metadata for a specific chunk.

        Args:
            chunk_index: Index of chunk

        Returns:
            Dictionary with chunk metadata
        """
        if chunk_index >= len(self.last_chunks):
            return {}

        chunk = self.last_chunks[chunk_index]
        return {
            'chunk_id': chunk.chunk_id,
            'token_count': chunk.token_count,
            'page_no': chunk.page_no,
            'section_title': chunk.section_title,
            'is_table': chunk.is_table,
            'is_heading': chunk.is_heading,
            'bbox': chunk.bbox,
            **chunk.metadata
        }
```

### 4. Unified Multimodal Loader

Handle all file types with a single loader:

**Implementation:**
```python
from neo4j_graphrag.experimental.components.pdf_loader import DataLoader
from vertector_data_ingestion import (
    UniversalConverter,
    create_audio_transcriber,
    LocalMpsConfig,
    AudioConfig,
    WhisperModelSize,
    ExportFormat,
)
from pathlib import Path

class MultimodalLoader(DataLoader):
    """Unified loader for documents and audio files."""

    def __init__(self, vertector_config=None, audio_config=None):
        """
        Initialize multimodal loader.

        Args:
            vertector_config: Config for documents (LocalMpsConfig, CloudGpuConfig, etc.)
            audio_config: AudioConfig for audio files
        """
        # Document converter
        self.converter = UniversalConverter(vertector_config or LocalMpsConfig())

        # Audio transcriber
        if audio_config is None:
            audio_config = AudioConfig(model_size=WhisperModelSize.BASE)
        self.transcriber = create_audio_transcriber(audio_config)

        self.last_metadata = {}

    async def load(self, path: str) -> str:
        """
        Load any supported file type.

        Args:
            path: Path to file (document or audio)

        Returns:
            Markdown text representation
        """
        path = Path(path)

        # Detect file type
        if path.suffix.lower() in ['.wav', '.mp3', '.m4a', '.flac', '.ogg']:
            return await self._load_audio(path)
        else:
            return await self._load_document(path)

    async def _load_document(self, path: Path) -> str:
        """Load document file."""
        doc = self.converter.convert(path)
        text = self.converter.export(doc, ExportFormat.MARKDOWN)

        self.last_metadata = {
            'modality': 'document',
            'source_path': str(path),
            'num_pages': doc.metadata.num_pages,
            'pipeline_type': doc.metadata.pipeline_type,
        }

        return text

    async def _load_audio(self, path: Path) -> str:
        """Load and transcribe audio file."""
        result = self.transcriber.transcribe(path)

        # Format transcript
        parts = [f"# {path.name} Transcript\n"]
        for segment in result.segments:
            parts.append(f"[{segment.start:.1f}s] {segment.text}")

        text = "\n".join(parts)

        self.last_metadata = {
            'modality': 'audio',
            'source_path': str(path),
            'duration': result.duration,
            'language': result.language,
        }

        return text
```

## Integration Approaches

### Approach 1: Component Replacement (Recommended)

Replace specific SimpleKGPipeline components:

```python
from neo4j_graphrag.experimental.pipeline import SimpleKGPipeline
from vertector_data_ingestion import LocalMpsConfig

# Initialize Vertector components
loader = MultimodalLoader(
    vertector_config=LocalMpsConfig(),  # or CloudGpuConfig() or CloudCpuConfig()
)

splitter = VertectorTextSplitter(
    chunk_size=512,
    tokenizer="Qwen/Qwen3-Embedding-0.6B"
)

# Create SimpleKGPipeline with custom components
pipeline = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedder,
    # Custom Vertector components
    pdf_loader=loader,
    text_splitter=splitter,
    # Neo4j components
    on_error="RAISE"
)

# Process documents
await pipeline.run_async(file_path="document.pdf")

# Process audio
await pipeline.run_async(file_path="meeting.wav")
```

**Pros:**
- Clean separation of concerns
- Leverages both libraries' strengths
- Hardware-optimized via config classes
- Supports all Vertector modalities

**Cons:**
- Requires implementing adapter interfaces
- Some metadata may be lost in text conversion

### Approach 2: Pre-Processing Pipeline

Use Vertector as pre-processing step:

```python
from vertector_data_ingestion import (
    UniversalConverter,
    HybridChunker,
    LocalMpsConfig,
    ExportFormat,
)
from vertector_data_ingestion.models.config import ChunkingConfig
from neo4j_graphrag.experimental.pipeline import SimpleKGPipeline

# Stage 1: Vertector ingestion with config
config = LocalMpsConfig()  # Hardware-optimized
converter = UniversalConverter(config)
doc = converter.convert("document.pdf")

# Stage 2: Chunking
chunker = HybridChunker(
    config=ChunkingConfig(
        tokenizer="Qwen/Qwen3-Embedding-0.6B",
        max_tokens=512
    )
)
chunks = chunker.chunk_document(doc)

# Stage 3: Export for Neo4j
markdown_text = converter.export(doc, ExportFormat.MARKDOWN)

# Stage 4: Neo4j knowledge graph construction
pipeline = SimpleKGPipeline(llm=llm, driver=driver)
await pipeline.run_async(text=markdown_text)

# Optional: Enrich graph with chunk metadata
for chunk in chunks.chunks:
    # Add metadata to Neo4j nodes
    await add_chunk_metadata(chunk)
```

**Pros:**
- No modification to Neo4j components
- Full control over each stage
- Access to all Vertector metadata

**Cons:**
- Less integrated
- Manual metadata management

### Approach 3: Unified Pipeline

Create orchestrator class:

```python
from vertector_data_ingestion import (
    UniversalConverter,
    HybridChunker,
    create_audio_transcriber,
    LocalMpsConfig,
    AudioConfig,
    WhisperModelSize,
)
from neo4j_graphrag.experimental.pipeline import SimpleKGPipeline

class VertectorNeo4jPipeline:
    """Unified multimodal knowledge graph pipeline."""

    def __init__(
        self,
        vertector_config=None,
        neo4j_driver=None,
        llm=None,
        embedder=None,
    ):
        # Vertector components
        self.converter = UniversalConverter(vertector_config or LocalMpsConfig())
        self.chunker = HybridChunker()
        self.transcriber = create_audio_transcriber(
            AudioConfig(model_size=WhisperModelSize.BASE)
        )

        # Neo4j pipeline with custom loaders
        self.kg_pipeline = SimpleKGPipeline(
            llm=llm,
            driver=neo4j_driver,
            embedder=embedder,
            pdf_loader=MultimodalLoader(),
            text_splitter=VertectorTextSplitter(),
        )

    async def process(self, file_path: str) -> dict:
        """
        Process any file type: documents or audio.

        Args:
            file_path: Path to file

        Returns:
            Processing results with metadata
        """
        path = Path(file_path)

        # Run through Neo4j pipeline
        kg_result = await self.kg_pipeline.run_async(file_path=str(path))

        return {
            'file': path.name,
            'kg_result': kg_result,
            'metadata': self.kg_pipeline.pdf_loader.last_metadata
        }
```

## Configuration

### Hardware-Optimized Configurations

```python
from vertector_data_ingestion import LocalMpsConfig, CloudGpuConfig, CloudCpuConfig

# Apple Silicon (M1/M2/M3/M4)
mps_config = LocalMpsConfig()
pipeline = MultimodalLoader(vertector_config=mps_config)

# NVIDIA GPU
gpu_config = CloudGpuConfig()
pipeline = MultimodalLoader(vertector_config=gpu_config)

# CPU-only
cpu_config = CloudCpuConfig()
pipeline = MultimodalLoader(vertector_config=cpu_config)
```

### Custom Configuration

```python
from vertector_data_ingestion.models.config import (
    ConverterConfig,
    VlmConfig,
    AudioConfig,
    ChunkingConfig,
)

custom_config = ConverterConfig(
    vlm=VlmConfig(
        use_mlx=True,
        preset_model="granite-mlx",
    ),
    audio=AudioConfig(
        model_size=WhisperModelSize.BASE,
        backend=AudioBackend.AUTO,
    ),
    chunking=ChunkingConfig(
        tokenizer="Qwen/Qwen3-Embedding-0.6B",
        max_tokens=512,
    ),
)

pipeline = MultimodalLoader(vertector_config=custom_config)
```

## Use Cases

### 1. Research Paper Knowledge Graphs

```python
from vertector_data_ingestion import UniversalConverter, LocalMpsConfig

converter = UniversalConverter(LocalMpsConfig())
pipeline = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    pdf_loader=VertectorDataLoader(LocalMpsConfig()),
    text_splitter=VertectorTextSplitter(chunk_size=512),
)

# Process paper
await pipeline.run_async(file_path="research_paper.pdf")

# Graph includes:
# - Entities: Authors, Methods, Findings, Concepts
# - Relationships: AUTHORED_BY, USES_METHOD, CITES
# - Metadata: Page numbers, sections, tables, figures
```

### 2. Meeting Transcripts with Context

```python
from vertector_data_ingestion import create_audio_transcriber, AudioConfig

# Transcribe meeting
audio_loader = VertectorAudioLoader(
    AudioConfig(model_size=WhisperModelSize.BASE, word_timestamps=True)
)

pipeline = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    pdf_loader=audio_loader,
)

await pipeline.run_async(file_path="meeting.wav")

# Graph includes:
# - Entities: People, Topics, Action Items
# - Relationships: DISCUSSED, ASSIGNED_TO
# - Metadata: Timestamps, speakers, duration
```

### 3. Multimodal Corporate Documents

```python
# Process presentation with slides and audio
docs = ["quarterly_report.pptx", "earnings_call.mp3"]

for doc_path in docs:
    await pipeline.run_async(file_path=doc_path)

# Unified graph with:
# - Document content + audio transcript
# - Cross-references between slides and discussion
# - Temporal alignment via timestamps
```

## Testing

### Unit Tests

```python
import pytest
from vertector_data_ingestion import LocalMpsConfig

@pytest.mark.asyncio
async def test_document_loader():
    loader = VertectorDataLoader(LocalMpsConfig())
    text = await loader.load("test.pdf")

    assert text is not None
    assert len(text) > 0
    assert loader.last_metadata['filename'] == 'test.pdf'

@pytest.mark.asyncio
async def test_audio_loader():
    loader = VertectorAudioLoader()
    text = await loader.load("test.wav")

    assert 'Transcription' in text
    assert loader.last_metadata['modality'] == 'audio'

def test_text_splitter():
    splitter = VertectorTextSplitter(chunk_size=512)
    chunks = await splitter.split("Long text here...")

    assert isinstance(chunks, list)
    assert all(isinstance(c, str) for c in chunks)
```

## Performance Optimization

### Batch Processing

```python
from pathlib import Path

# Batch document processing
doc_paths = list(Path("documents/").glob("*.pdf"))
docs = converter.convert(doc_paths, parallel=True)

for doc in docs:
    text = converter.export(doc, ExportFormat.MARKDOWN)
    await pipeline.run_async(text=text)
```

### Caching

```python
from vertector_data_ingestion.models.config import ConverterConfig

config = ConverterConfig(
    enable_cache=True,
    cache_dir=Path("~/.vertector/cache"),
)
converter = UniversalConverter(config)
```

## Troubleshooting

### Issue: Metadata Lost

**Problem:** Rich metadata from Vertector not in Neo4j graph.

**Solution:** Use custom writer to preserve metadata:

```python
from neo4j_graphrag.experimental.components.kg_writer import Neo4jWriter

class EnrichedNeo4jWriter(Neo4jWriter):
    async def write_chunk(self, chunk_text: str, metadata: dict):
        # Add Vertector metadata to nodes
        query = """
        MERGE (c:Chunk {text: $text})
        SET c.page_no = $page_no,
            c.section = $section,
            c.token_count = $token_count
        """
        await self.driver.execute_query(query, text=chunk_text, **metadata)
```

### Issue: Audio Chunks Too Long

**Problem:** Audio transcripts create very long chunks.

**Solution:** Use time-based chunking:

```python
def chunk_by_time(segments, max_duration=30):
    """Group segments by time window."""
    chunks = []
    current = []
    current_duration = 0

    for seg in segments:
        duration = seg.end - seg.start
        if current_duration + duration > max_duration and current:
            chunks.append(" ".join(s.text for s in current))
            current = []
            current_duration = 0
        current.append(seg)
        current_duration += duration

    if current:
        chunks.append(" ".join(s.text for s in current))

    return chunks
```

## References

### Neo4j Documentation
- [User Guide: Knowledge Graph Builder](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html)
- [GitHub: neo4j-graphrag-python](https://github.com/neo4j/neo4j-graphrag-python)
- [API Documentation](https://neo4j.com/docs/neo4j-graphrag-python/current/api.html)
- [Chunking | GraphAcademy](https://www.graphacademy.neo4j.com/courses/llm-vectors-unstructured/3-unstructured-data/2-chunking/)

### Vertector Documentation
- [Installation Guide](installation.md)
- [Configuration Guide](configuration.md)
- [MCP Server Guide](mcp-server.md)

### Notebook Examples
- [Document Processing](../notebooks/01_document_processing.ipynb)
- [Audio Transcription](../notebooks/02_audio_transcription.ipynb)
- [RAG Pipeline](../notebooks/03_rag_pipeline.ipynb)
- [Multimodal Integration](../notebooks/04_multimodal_integration.ipynb)

## Next Steps

1. **Prototype**: Implement basic VertectorDataLoader
2. **Test**: Validate with sample documents and audio
3. **Extend**: Add metadata enrichment writer
4. **Package**: Create `vertector-neo4j` integration package
5. **Document**: Add comprehensive examples
6. **Release**: Publish to PyPI

---

**Status**: Design Phase
**Last Updated**: 2026-01-02
**Contributors**: Enoch Tetteh, Claude Code
