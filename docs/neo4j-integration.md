# Neo4j SimpleKGPipeline Integration

Integration design for Vertector Data Ingestion with Neo4j's SimpleKGPipeline for end-to-end knowledge graph construction.

## Overview

This integration combines Vertector's multimodal data ingestion capabilities with Neo4j's knowledge graph construction, creating a powerful pipeline that can:

1. **Ingest multimodal content** (documents, images, audio) using Vertector
2. **Extract structured knowledge** using Neo4j's SimpleKGPipeline
3. **Build knowledge graphs** with rich metadata and relationships

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    End-to-End KG Pipeline                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────┐      ┌──────────────────────┐       │
│  │  Vertector Ingestion │─────▶│ Neo4j SimpleKGPipeline│       │
│  └──────────────────────┘      └──────────────────────┘       │
│           │                              │                      │
│           ▼                              ▼                      │
│  ┌──────────────────────┐      ┌──────────────────────┐       │
│  │  Multimodal Processing│      │  Knowledge Extraction │       │
│  │  - PDF, DOCX, PPTX   │      │  - Entity Recognition │       │
│  │  - Images (VLM)      │      │  - Relation Extraction│       │
│  │  - Audio (Whisper)   │      │  - Schema Validation  │       │
│  │  - Tables & Metadata │      │  - Entity Resolution  │       │
│  └──────────────────────┘      └──────────────────────┘       │
│           │                              │                      │
│           └──────────────┬───────────────┘                     │
│                          ▼                                      │
│                 ┌──────────────────┐                           │
│                 │   Neo4j Graph    │                           │
│                 │    Database      │                           │
│                 └──────────────────┘                           │
└─────────────────────────────────────────────────────────────────┘
```

## Integration Points

### 1. Custom Data Loader (Primary Integration)

Replace Neo4j's default `PdfLoader` with Vertector's `UniversalConverter`:

**Benefits:**
- Support for multiple formats beyond PDF (DOCX, PPTX, XLSX, HTML)
- Vision-Language Model pipeline for images
- Audio transcription integration
- Advanced table extraction
- Rich metadata preservation

**Implementation:**
```python
from neo4j_graphrag.experimental.components.pdf_loader import DataLoader
from vertector_data_ingestion import UniversalConverter, ExportFormat

class VertectorDataLoader(DataLoader):
    """Custom data loader using Vertector for multimodal ingestion."""

    def __init__(self, config=None):
        self.converter = UniversalConverter(config)

    async def load(self, path: str) -> str:
        """Load document using Vertector."""
        doc_wrapper = self.converter.convert_single(path)

        # Export to markdown for text processing
        text = self.converter.export(doc_wrapper, ExportFormat.MARKDOWN)

        # Preserve metadata for graph enrichment
        self.metadata = {
            'source_path': str(path),
            'num_pages': doc_wrapper.metadata.num_pages,
            'pipeline_type': doc_wrapper.metadata.pipeline_type,
            'processing_time': doc_wrapper.metadata.processing_time,
        }

        return text
```

### 2. Enhanced Text Splitter

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

class VertectorTextSplitter(TextSplitter):
    """Text splitter using Vertector's HybridChunker."""

    def __init__(self, chunk_size: int = 512, tokenizer: str = "Qwen/Qwen3-Embedding-0.6B"):
        config = ChunkingConfig(
            tokenizer=tokenizer,
            max_tokens=chunk_size,
        )
        self.chunker = HybridChunker(config=config)

    async def split(self, text: str) -> list[str]:
        """Split text using hybrid chunking strategy."""
        # Create minimal document wrapper for chunking
        from vertector_data_ingestion.models.document import DoclingDocumentWrapper

        # Use chunker's text chunking capability
        chunking_result = self.chunker.chunk_text(text)

        # Return chunk texts with preserved metadata
        chunks = []
        for chunk in chunking_result.chunks:
            # Optionally enrich with metadata
            chunk_text = f"{chunk.text}\n\n[Page: {chunk.page_number}, Section: {' > '.join(chunk.section_hierarchy)}]"
            chunks.append(chunk_text)

        return chunks
```

### 3. Metadata Enrichment Writer

Extend Neo4j's writer to include Vertector metadata:

**Benefits:**
- Document-level metadata (processing pipeline, quality metrics)
- Chunk-level metadata (page numbers, sections, bounding boxes)
- Visual grounding information from VLM pipeline
- Audio transcription metadata (timestamps, confidence)

**Implementation:**
```python
from neo4j_graphrag.experimental.components.kg_writer import KGWriter

class EnrichedNeo4jWriter(KGWriter):
    """Neo4j writer with Vertector metadata enrichment."""

    def __init__(self, driver, vertector_metadata: dict):
        super().__init__(driver)
        self.vertector_metadata = vertector_metadata

    async def write_document_node(self, doc_id: str, text: str):
        """Write document node with enriched metadata."""
        query = """
        MERGE (d:Document {id: $doc_id})
        SET d.text = $text,
            d.source_path = $source_path,
            d.num_pages = $num_pages,
            d.pipeline_type = $pipeline_type,
            d.processing_time = $processing_time,
            d.ingestion_timestamp = timestamp()
        """
        await self.driver.execute_query(
            query,
            doc_id=doc_id,
            text=text,
            **self.vertector_metadata
        )

    async def write_chunk_node(self, chunk_id: str, chunk_data: dict):
        """Write chunk with spatial and hierarchical metadata."""
        query = """
        MERGE (c:Chunk {id: $chunk_id})
        SET c.text = $text,
            c.page_number = $page_number,
            c.section_hierarchy = $section_hierarchy,
            c.bbox = $bbox,
            c.chunk_index = $chunk_index
        """
        # Include bounding box for visual grounding
        bbox_str = None
        if chunk_data.get('bbox'):
            bbox = chunk_data['bbox']
            bbox_str = f"[{bbox.l},{bbox.t},{bbox.r},{bbox.b}]"

        await self.driver.execute_query(
            query,
            chunk_id=chunk_id,
            text=chunk_data['text'],
            page_number=chunk_data.get('page_number'),
            section_hierarchy=chunk_data.get('section_hierarchy', []),
            bbox=bbox_str,
            chunk_index=chunk_data.get('chunk_index', 0)
        )
```

## Integration Approaches

### Approach 1: Component Replacement (Recommended)

Replace specific SimpleKGPipeline components with Vertector equivalents:

```python
from neo4j_graphrag.experimental.pipeline import SimpleKGPipeline
from vertector_data_ingestion import UniversalConverter, HybridChunker

# Initialize Vertector components
vertector_loader = VertectorDataLoader()
vertector_splitter = VertectorTextSplitter(chunk_size=512)

# Create SimpleKGPipeline with custom components
pipeline = SimpleKGPipeline(
    llm=llm,
    driver=driver,
    embedder=embedder,
    # Custom Vertector components
    pdf_loader=vertector_loader,
    text_splitter=vertector_splitter,
    # Standard Neo4j components
    kg_writer=kg_writer,
    on_error="RAISE"
)

# Process documents
await pipeline.run_async(file_path="multimodal_document.pdf")
```

**Pros:**
- Clean separation of concerns
- Leverages both libraries' strengths
- Minimal code changes to existing pipelines
- Easy to maintain and test

**Cons:**
- Requires implementing adapter interfaces
- May have some data transformation overhead

### Approach 2: Pre-Processing Pipeline

Use Vertector as a pre-processing step before Neo4j:

```python
from vertector_data_ingestion import UniversalConverter, HybridChunker
from neo4j_graphrag.experimental.pipeline import SimpleKGPipeline

# Stage 1: Vertector ingestion
converter = UniversalConverter()
doc = converter.convert_single("document.pdf")

# Stage 2: Chunking
chunker = HybridChunker()
chunks = chunker.chunk_document(doc)

# Stage 3: Export for Neo4j
markdown_text = converter.export(doc, ExportFormat.MARKDOWN)

# Stage 4: Neo4j knowledge graph construction
pipeline = SimpleKGPipeline(llm=llm, driver=driver)
await pipeline.run_async(text=markdown_text)
```

**Pros:**
- No modification to Neo4j components
- Full control over each stage
- Easy to debug and monitor

**Cons:**
- Less integrated
- Potential data duplication
- Metadata may be lost between stages

### Approach 3: Unified Pipeline (Most Integrated)

Create a new unified pipeline class that orchestrates both:

```python
class VertectorNeo4jPipeline:
    """Unified pipeline combining Vertector ingestion with Neo4j KG construction."""

    def __init__(
        self,
        vertector_config=None,
        neo4j_driver=None,
        llm=None,
        embedder=None,
    ):
        self.converter = UniversalConverter(vertector_config)
        self.chunker = HybridChunker()
        self.kg_pipeline = SimpleKGPipeline(
            llm=llm,
            driver=neo4j_driver,
            embedder=embedder,
            pdf_loader=VertectorDataLoader(),
            text_splitter=VertectorTextSplitter(),
        )

    async def process_document(self, file_path: str) -> dict:
        """End-to-end processing: ingestion → chunking → KG construction."""
        # Stage 1: Multimodal ingestion
        doc = self.converter.convert_single(file_path)

        # Stage 2: Intelligent chunking
        chunks = self.chunker.chunk_document(doc)

        # Stage 3: Knowledge graph construction
        kg_result = await self.kg_pipeline.run_async(
            text=self.converter.export(doc, ExportFormat.MARKDOWN)
        )

        # Stage 4: Metadata enrichment
        await self._enrich_graph_with_metadata(doc, chunks, kg_result)

        return {
            'document': doc,
            'chunks': chunks,
            'kg_result': kg_result,
        }

    async def _enrich_graph_with_metadata(self, doc, chunks, kg_result):
        """Add Vertector metadata to Neo4j graph."""
        # Implementation details...
        pass
```

**Pros:**
- Seamless integration
- Single interface for entire pipeline
- Maximum metadata preservation
- Best user experience

**Cons:**
- Most complex to implement
- Tighter coupling between systems
- Requires more maintenance

## Use Cases

### 1. Multimodal Document Analysis

**Scenario:** Process corporate documents containing text, tables, images, and diagrams.

**Pipeline:**
1. Vertector extracts all modalities (text, tables, images via VLM)
2. HybridChunker creates semantically coherent chunks
3. Neo4j extracts entities and relationships
4. Knowledge graph includes visual elements as nodes

**Example Graph:**
```cypher
(Report:Document)-[:CONTAINS]->(Section:TextChunk)
(Section)-[:DESCRIBES]->(Product:Entity)
(Report)-[:HAS_TABLE]->(FinancialData:Table)
(Report)-[:HAS_IMAGE]->(Diagram:Image {description: "VLM extracted"})
```

### 2. Research Paper Knowledge Graphs

**Scenario:** Build knowledge graphs from academic papers with citations, figures, and equations.

**Pipeline:**
1. Vertector processes PDF with table/image extraction
2. Preserve section hierarchy (Abstract, Methods, Results)
3. Neo4j extracts research concepts, methodologies, findings
4. Link papers through citations

### 3. Multimedia Content Processing

**Scenario:** Process presentations with slides, speaker notes, and audio.

**Pipeline:**
1. Vertector processes PPTX (slides + notes)
2. Transcribe audio using Whisper
3. Align transcript with slides using timestamps
4. Build knowledge graph with temporal relationships

### 4. Legal Document Analysis

**Scenario:** Extract entities and relationships from contracts and legal documents.

**Pipeline:**
1. Vertector extracts text, tables (terms, clauses)
2. Preserve page numbers and bounding boxes for citations
3. Neo4j identifies parties, obligations, dates
4. Link related clauses and references

## Implementation Roadmap

### Phase 1: Core Integration (Week 1-2)
- [ ] Implement `VertectorDataLoader` adapter
- [ ] Implement `VertectorTextSplitter` adapter
- [ ] Create basic integration tests
- [ ] Document component replacement approach

### Phase 2: Metadata Enrichment (Week 3-4)
- [ ] Extend Neo4j writer with metadata support
- [ ] Add visual grounding (bounding boxes) to graph
- [ ] Implement section hierarchy preservation
- [ ] Add audio transcription metadata

### Phase 3: Unified Pipeline (Week 5-6)
- [ ] Create `VertectorNeo4jPipeline` class
- [ ] Implement end-to-end processing
- [ ] Add batch processing support
- [ ] Create comprehensive examples

### Phase 4: Advanced Features (Week 7-8)
- [ ] Multi-document knowledge graph construction
- [ ] Cross-document entity resolution
- [ ] Temporal relationship tracking
- [ ] Visual similarity clustering

## Configuration Example

```yaml
# config.yaml
vertector:
  vlm:
    use_mlx: true
    preset_model: "granite-mlx"
  audio:
    backend: "mlx"
    model_size: "base"
  chunking:
    tokenizer: "Qwen/Qwen3-Embedding-0.6B"
    max_tokens: 512

neo4j:
  uri: "bolt://localhost:7687"
  database: "knowledge-graph"
  schema: "EXTRACTED"
  perform_entity_resolution: true

pipeline:
  batch_size: 10
  on_error: "RAISE"
  export_format: "markdown"
```

## API Design

### Simplified Interface

```python
from vertector_neo4j import create_kg_pipeline

# Initialize pipeline
pipeline = create_kg_pipeline(
    vertector_config="config.yaml",
    neo4j_uri="bolt://localhost:7687",
    llm=llm,
)

# Process single document
result = await pipeline.process("document.pdf")

# Process batch
results = await pipeline.process_batch([
    "paper1.pdf",
    "presentation.pptx",
    "meeting_audio.wav",
])

# Query knowledge graph
entities = await pipeline.query_entities(label="Person")
relationships = await pipeline.query_relationships(type="WORKS_FOR")
```

## Benefits of Integration

### 1. Enhanced Data Ingestion
- **Multi-format support**: Beyond PDFs to DOCX, PPTX, XLSX, images, audio
- **Better quality**: VLM pipeline for images, advanced table extraction
- **Rich metadata**: Page numbers, sections, bounding boxes, timestamps

### 2. Improved Chunking
- **Token-aware**: Respects LLM context windows
- **Semantic coherence**: Preserves document structure
- **Hierarchical**: Maintains section relationships

### 3. Knowledge Graph Quality
- **Visual grounding**: Link text to image regions
- **Temporal data**: Audio transcription timestamps
- **Structural metadata**: Section hierarchy preservation
- **Source tracking**: Precise citations with page/bbox

### 4. Flexibility
- **Hardware optimization**: Auto-detect MPS/CUDA/CPU
- **Configurable pipelines**: Mix and match components
- **Extensible**: Easy to add custom processors

## Challenges and Solutions

### Challenge 1: Data Format Conversion

**Problem:** Neo4j expects text strings, Vertector produces rich structured data.

**Solution:** Create flexible exporters that preserve metadata in embedded format:
```markdown
## Introduction [page:1, section:1.0]

This is the introduction text...

![Diagram](image_id_123) <!-- VLM: "Architecture diagram showing three layers" -->
```

### Challenge 2: Metadata Preservation

**Problem:** Neo4j's default pipeline doesn't preserve spatial/visual metadata.

**Solution:** Extend graph schema to include Vertector metadata:
```cypher
CREATE (c:Chunk {
    text: "...",
    page: 5,
    bbox: [100, 200, 500, 300],
    section: ["Chapter 3", "Methods"],
    embedding: [0.1, 0.2, ...]
})
```

### Challenge 3: Performance Overhead

**Problem:** Processing multiple modalities may slow down pipeline.

**Solution:**
- Parallel processing of chunks
- Async/await throughout
- Caching of converted documents
- Batch writes to Neo4j

### Challenge 4: Schema Compatibility

**Problem:** Different data types need compatible graph schemas.

**Solution:** Create modality-specific node types:
```cypher
(:TextChunk)-[:NEXT]->(:TextChunk)
(:Image {vlm_description})-[:APPEARS_IN]->(:TextChunk)
(:Table {rows, columns})-[:REFERENCED_BY]->(:Entity)
(:AudioSegment {start, end, text})-[:TRANSCRIBES]->(:Document)
```

## Testing Strategy

### Unit Tests
- Test each adapter component independently
- Mock Neo4j driver and Vertector components
- Verify metadata transformation

### Integration Tests
- Test complete pipeline with real documents
- Verify graph structure and relationships
- Test error handling and recovery

### Performance Tests
- Benchmark processing speed vs. defaults
- Memory usage profiling
- Scalability testing with large documents

## Documentation Requirements

1. **Installation Guide**: Dependencies, setup, configuration
2. **Quick Start**: Basic usage examples
3. **API Reference**: Complete API documentation
4. **Architecture Guide**: Design decisions, component interaction
5. **Examples**: Real-world use cases with code
6. **Migration Guide**: From default Neo4j pipeline

## Success Metrics

- **Coverage**: % of document types successfully processed
- **Quality**: Precision/recall of entity extraction
- **Performance**: Documents processed per hour
- **Metadata**: % of metadata preserved in graph
- **User adoption**: GitHub stars, PyPI downloads

## References

### Neo4j Documentation
- [User Guide: Knowledge Graph Builder](https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html)
- [GitHub: neo4j-graphrag-python](https://github.com/neo4j/neo4j-graphrag-python)
- [SimpleKGPipeline API](https://neo4j.com/docs/neo4j-graphrag-python/current/_modules/neo4j_graphrag/experimental/pipeline/config/template_pipeline/simple_kg_builder.html)

### Vertector Documentation
- [Installation Guide](installation.md)
- [User Guide](user-guide.md)
- [MCP Server](mcp-server.md)
- [Configuration](configuration.md)

## Next Steps

1. **Prototype**: Build minimal viable integration (Approach 1)
2. **Validate**: Test with real documents from target use cases
3. **Iterate**: Gather feedback and refine design
4. **Package**: Create `vertector-neo4j` integration package
5. **Document**: Write comprehensive guides and examples
6. **Release**: Publish to PyPI and announce to community

---

**Status**: Design Phase
**Last Updated**: 2026-01-02
**Contributors**: Enoch Tetteh, Claude Code
