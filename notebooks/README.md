# Vertector Data Ingestion - Jupyter Notebooks

Interactive tutorials demonstrating all data ingestion modalities and features.

## Notebooks Overview

### 1. Document Processing (`01_document_processing.ipynb`)

Comprehensive document processing tutorial.

**Topics Covered**:
- Hardware detection and optimization
- Basic document conversion (PDF, DOCX, PPTX, XLSX)
- Export to multiple formats (Markdown, JSON, DocTags)
- Classic vs VLM pipeline comparison
- Batch processing
- Table extraction
- OCR configuration
- Multi-format support

**Best For**: Getting started with document processing

### 2. Audio Transcription (`02_audio_transcription.ipynb`)

Complete audio transcription guide.

**Topics Covered**:
- Whisper model configuration
- Hardware acceleration (MLX vs Standard)
- Timestamped transcriptions
- Model size comparison (speed vs accuracy)
- Multi-language support
- High-accuracy configuration
- Batch audio processing
- SRT subtitle generation
- Backend performance comparison

**Best For**: Audio-to-text conversion workflows

### 3. RAG Pipeline (`03_rag_pipeline.ipynb`)

Complete RAG implementation from end to end.

**Topics Covered**:
- Basic RAG pipeline (Document → Chunks → Vector Store → Search)
- Chunking strategies (small, medium, large)
- Chunk inspection and metadata
- Batch document ingestion
- Advanced search techniques
- RAG context assembly for LLMs
- Export and analysis
- Performance optimization tips

**Best For**: Building RAG applications

### 4. Multimodal Integration (`04_multimodal_integration.ipynb`)

Combining all data modalities in a unified system.

**Topics Covered**:
- Unified multimodal pipeline
- Processing documents and audio together
- Cross-modal semantic search
- Modality-specific filtering
- Production workflow examples (meeting knowledge base)
- Analytics and insights
- Export and backup strategies

**Best For**: Production multimodal applications

## Running the Notebooks

### Prerequisites

Make sure the package is installed in development mode:

```bash
# Install the package in editable mode
uv pip install -e .

# Verify installation
uv pip list | grep vertector-data-ingestion
```

### Setup

```bash
# Install Jupyter (only needed once)
uv add jupyter

# Start Jupyter with the correct Python environment
uv run jupyter notebook

# Or use JupyterLab
uv add jupyterlab
uv run jupyter lab
```

**Important**: Always use `uv run jupyter notebook` or `uv run jupyter lab` to ensure Jupyter uses the correct Python environment with the installed package.

### In VS Code

1. Install Python extension
2. Open notebook file (.ipynb)
3. Select the correct Python interpreter (the one from your uv environment)
   - Press `Cmd+Shift+P` (Mac) or `Ctrl+Shift+P` (Windows/Linux)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from `.venv` directory
4. Run cells

### In Google Colab

1. Upload notebook to Google Drive
2. Open with Google Colab
3. Install dependencies:
   ```python
   !pip install vertector-data-ingestion
   ```

## Quick Start

### Minimal Example

```python
# Import
from vertector_data_ingestion import UniversalConverter

# Convert
converter = UniversalConverter()
doc = converter.convert_single("document.pdf")

# Use
print(doc.export_to_markdown())
```

## Notebook Structure

Each notebook follows this structure:

1. **Setup and Imports** - Load required packages
2. **Basic Examples** - Simple, working code
3. **Configuration** - Customization options
4. **Advanced Topics** - Production features
5. **Summary** - Key takeaways and next steps

## Prerequisites

### Required Files

Some notebooks expect sample files:
- `path/to/your/document.pdf` - Any PDF document
- `path/to/your/audio.wav` - Any audio file
- `documents/` - Directory with multiple documents
- `audio_files/` - Directory with audio files

### Optional Dependencies

Depending on features used:

```bash
# For audio (Apple Silicon)
uv add mlx-whisper

# For audio (CUDA/CPU)
uv add openai-whisper

# For OCR
uv add easyocr pytesseract
```

## Common Workflows

### Research Paper Processing

```python
# Convert paper
from vertector_data_ingestion import UniversalConverter, HybridChunker

converter = UniversalConverter()
doc = converter.convert_single("research_paper.pdf")

# Create chunks for RAG
chunker = HybridChunker()
chunks = chunker.chunk_document(doc.document)
```

### Meeting Transcription

```python
# Transcribe meeting
from vertector_data_ingestion import create_audio_transcriber, AudioConfig

transcriber = create_audio_transcriber(AudioConfig())
result = transcriber.transcribe("meeting.wav")

# Get timestamped segments
for segment in result.segments:
    print(f"[{segment.start:.1f}s] {segment.text}")
```

### Knowledge Base Search

```python
# Build knowledge base
from vertector_data_ingestion import ChromaAdapter, HybridChunker

chunker = HybridChunker()
vector_store = ChromaAdapter(collection_name="kb")

# Add documents
docs = ["doc1.pdf", "doc2.pdf"]
for doc in docs:
    chunks = chunker.chunk_document(doc)
    vector_store.add_chunks(chunks.chunks)

# Search
results = vector_store.search("How does it work?", top_k=5)
```

## Troubleshooting

### Kernel Not Found

```bash
# Install ipykernel
uv add ipykernel

# Register kernel
uv run python -m ipykernel install --user --name=vertector
```

### Import Errors

If you see `ModuleNotFoundError: No module named 'vertector_data_ingestion'`:

```bash
# 1. Make sure package is installed in editable mode
uv pip install -e .

# 2. Verify installation
uv pip list | grep vertector-data-ingestion

# 3. Restart Jupyter using uv run
uv run jupyter notebook

# 4. In Jupyter: Restart kernel
# Kernel > Restart
```

**Important**: The most common cause is not using `uv run` to start Jupyter. This ensures Jupyter uses the correct Python environment with the installed package.

### Out of Memory

```python
# Use smaller models
config = ConverterConfig()
config.vlm.batch_size = 2  # Reduce batch size
config.audio.model_size = WhisperModelSize.TINY
```

## Tips and Best Practices

1. **Start Small**: Begin with a single file before batch processing
2. **Check Hardware**: Use hardware detection to optimize settings
3. **Save Outputs**: Export results to avoid reprocessing
4. **Use Caching**: Enable caching for faster iteration
5. **Monitor Memory**: Close unused notebooks to free memory

## Next Steps

After completing the notebooks:

1. Review [Documentation](../docs/README.md) for detailed guides
2. Check [Examples](../examples/README.md) for production scripts
3. Read [API Reference](../docs/api-reference.md) for complete API
4. Explore [Configuration Guide](../docs/configuration.md) for optimization

## Support

- **Email**: cutetetteh@gmail.com
- **Documentation**: [docs/](../docs/)
- **Examples**: [examples/](../examples/)

## Contributing

Found an issue or want to add a notebook? See [CONTRIBUTING.md](../CONTRIBUTING.md).
