## MCP Server for Vertector Data Ingestion

The Vertector MCP (Model Context Protocol) server exposes document processing, chunking, and metadata enrichment capabilities as tools that AI assistants can use.

## Overview

The MCP server provides 13 specialized tools organized into four categories:

1. **Document Processing** - Convert and extract from documents
2. **Chunking** - Create RAG-ready chunks
3. **Audio** - Transcribe audio files
4. **Utilities** - Hardware detection and validation

## Installation

```bash
# Install with MCP dependencies
uv sync --extra mcp

# Or install all extras
uv sync --extra all
```

## Running the Server

### Option 1: Command Line

```bash
vertector-mcp
```

### Option 2: Python Module

```python
python -m vertector_data_ingestion.mcp.server
```

### Option 3: Claude Desktop Integration

Add to your Claude Desktop config (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "vertector": {
      "command": "uv",
      "args": [
        "--directory",
        "/path/to/vertector-data-ingestion",
        "run",
        "vertector-mcp"
      ]
    }
  }
}
```

## Available Tools

### Document Processing Tools

#### `convert_document`

Convert a document (PDF, DOCX, PPTX, XLSX, images) to structured format.

**Parameters:**
- `file_path` (required): Path to the document file
- `output_format` (optional): "markdown" | "json" | "doctags" (default: "markdown")
- `pipeline` (optional): "auto" | "classic" | "vlm" (default: "auto")
- `hardware` (optional): "auto" | "mps" | "cuda" | "cpu" (default: "auto")

**Returns:**
```json
{
  "success": true,
  "content": "# Document Title\n\n...",
  "metadata": {
    "source_path": "/path/to/doc.pdf",
    "num_pages": 10,
    "file_size": 2048576,
    "format": "markdown",
    "pipeline_used": "auto",
    "hardware_used": "mps"
  }
}
```

**Example Usage:**
```
Convert the file research_paper.pdf to markdown format
```

---

#### `batch_convert_documents`

Convert multiple documents in parallel.

**Parameters:**
- `file_paths` (required): Array of document file paths
- `output_format`, `pipeline`, `hardware` (same as convert_document)
- `max_workers` (optional): Number of parallel workers (default: 4)

**Returns:**
```json
{
  "success": true,
  "total": 5,
  "successful": 4,
  "failed": 1,
  "results": [...]
}
```

**Example Usage:**
```
Convert all PDF files in the documents/ folder to JSON format
```

---

#### `extract_metadata`

Extract metadata from a document without full conversion (fast operation).

**Parameters:**
- `file_path` (required): Path to document file

**Returns:**
```json
{
  "success": true,
  "file_path": "/path/to/doc.pdf",
  "file_name": "doc.pdf",
  "file_size": 2048576,
  "file_size_mb": 2.0,
  "file_extension": ".pdf",
  "created": 1704067200.0,
  "modified": 1704153600.0
}
```

**Example Usage:**
```
Get metadata for presentation.pptx without converting it
```

---

#### `extract_tables`

Extract all tables from a document with structure preserved.

**Parameters:**
- `file_path` (required): Path to document file
- `output_format` (optional): "json" | "csv" | "markdown" (default: "json")
- `hardware` (optional): "auto" | "mps" | "cuda" | "cpu" (default: "auto")

**Returns:**
```json
{
  "success": true,
  "file_path": "/path/to/doc.pdf",
  "num_tables": 3,
  "tables": [...],
  "format": "json"
}
```

**Example Usage:**
```
Extract all tables from the financial_report.xlsx file
```

---

#### `extract_images`

Extract all images from a document (embedded images in PDFs, etc.).

**Parameters:**
- `file_path` (required): Path to document file
- `output_dir` (required): Directory to save extracted images
- `generate_captions` (optional): Use VLM to generate captions (default: false)
- `hardware` (optional): "auto" | "mps" | "cuda" | "cpu" (default: "auto")

**Returns:**
```json
{
  "success": true,
  "file_path": "/path/to/doc.pdf",
  "output_dir": "/path/to/images",
  "num_images": 5,
  "images": [
    {
      "index": 0,
      "path": "/path/to/images/image_0.png",
      "caption": "A diagram showing..."
    }
  ]
}
```

**Example Usage:**
```
Extract all images from technical_paper.pdf to the ./images folder and generate captions
```

---

### Chunking Tools

#### `chunk_document`

Create semantic chunks from a document for RAG applications.

**Parameters:**
- `file_path` (required): Path to the document file
- `max_tokens` (optional): Maximum tokens per chunk (default: 512)
- `overlap` (optional): Overlap tokens between chunks (default: 128)
- `tokenizer` (optional): HuggingFace tokenizer model (default: "Qwen/Qwen3-Embedding-0.6B")
- `include_metadata` (optional): Include rich metadata (default: true)
- `hardware` (optional): "auto" | "mps" | "cuda" | "cpu" (default: "auto")

**Returns:**
```json
{
  "success": true,
  "file_path": "/path/to/doc.pdf",
  "total_chunks": 25,
  "chunks": [
    {
      "chunk_id": "doc_0",
      "text": "Introduction\n\nThis paper presents...",
      "chunk_index": 0,
      "total_chunks": 25,
      "metadata": {
        "doc_id": "doc",
        "source_path": "/path/to/doc.pdf",
        "page_number": 1,
        "section_hierarchy": ["Introduction"],
        "bbox": {
          "left": 90.0,
          "top": 657.0,
          "right": 193.0,
          "bottom": 611.0
        }
      }
    }
  ],
  "statistics": {
    "total_chunks": 25,
    "min_chunk_size": 234,
    "max_chunk_size": 1024,
    "avg_chunk_size": 512.5
  },
  "config": {
    "max_tokens": 512,
    "overlap": 128,
    "tokenizer": "Qwen/Qwen3-Embedding-0.6B"
  }
}
```

**Example Usage:**
```
Chunk research_paper.pdf into 512-token chunks with 128 token overlap
```

---

#### `chunk_text`

Chunk raw text directly without file input.

**Parameters:**
- `text` (required): Raw text to chunk
- `max_tokens` (optional): Maximum tokens per chunk (default: 512)
- `overlap` (optional): Overlap tokens (default: 128)
- `tokenizer` (optional): HuggingFace tokenizer (default: "Qwen/Qwen3-Embedding-0.6B")
- `doc_id` (optional): Document ID for metadata

**Returns:**
```json
{
  "success": true,
  "total_chunks": 5,
  "chunks": [
    {
      "chunk_id": "text_0",
      "text": "...",
      "chunk_index": 0,
      "token_count": 512
    }
  ],
  "statistics": {...},
  "config": {...}
}
```

**Example Usage:**
```
Chunk this text into 256-token pieces: "Lorem ipsum dolor sit amet..."
```

---

#### `analyze_chunk_distribution`

Analyze chunk size distribution for a document to optimize chunking parameters.

**Parameters:**
- `file_path` (required): Path to document file
- `max_tokens`, `overlap`, `tokenizer`, `hardware` (same as chunk_document)

**Returns:**
```json
{
  "success": true,
  "file_path": "/path/to/doc.pdf",
  "total_chunks": 25,
  "distribution": {
    "min": 234,
    "max": 1024,
    "mean": 512.5,
    "median": 498,
    "p25": 450,
    "p75": 580,
    "p90": 650,
    "p99": 920
  },
  "histogram": {
    "ranges": [
      {"range": "0-100", "count": 0},
      {"range": "101-500", "count": 12},
      {"range": "501-1000", "count": 11},
      {"range": "1001-2000", "count": 2},
      {"range": "2001+", "count": 0}
    ]
  },
  "config": {...}
}
```

**Example Usage:**
```
Analyze the chunk distribution for document.pdf with 512 max tokens
```

---

### Audio Tools

#### `transcribe_audio`

Transcribe audio file to text using Whisper.

**Parameters:**
- `file_path` (required): Path to audio file
- `model_size` (optional): "tiny" | "base" | "small" | "medium" | "large" (default: "base")
- `language` (optional): Language code or "auto" (default: "auto")
- `include_timestamps` (optional): Include timestamps (default: true)
- `output_format` (optional): "text" | "srt" | "json" (default: "text")
- `backend` (optional): "auto" | "mlx" | "standard" (default: "auto")

**Returns:**
```json
{
  "success": true,
  "file_path": "/path/to/audio.wav",
  "transcription": "Hello, this is a test transcription...",
  "metadata": {
    "language": "en",
    "duration": 125.5,
    "num_segments": 15,
    "model_size": "base",
    "backend": "mlx"
  }
}
```

**Example Usage:**
```
Transcribe meeting.wav to SRT format with timestamps
```

---

#### `batch_transcribe_audio`

Transcribe multiple audio files in parallel.

**Parameters:**
- `file_paths` (required): Array of audio file paths
- `model_size`, `language`, `include_timestamps`, `output_format`, `backend` (same as transcribe_audio)
- `max_workers` (optional): Parallel workers (default: 2, audio is memory-intensive)

**Returns:**
```json
{
  "success": true,
  "total": 5,
  "successful": 4,
  "failed": 1,
  "results": [...]
}
```

**Example Usage:**
```
Transcribe all WAV files in the recordings/ folder
```

---

### Utility Tools

#### `detect_hardware`

Detect available hardware acceleration options and get recommendations.

**Parameters:** None

**Returns:**
```json
{
  "success": true,
  "available": {
    "mps": true,
    "cuda": false,
    "cpu": true
  },
  "cuda_devices": 0,
  "recommended": "mps",
  "recommendation_reason": "Apple Silicon MPS detected - optimal for VLM and audio"
}
```

**Example Usage:**
```
What hardware acceleration is available on this system?
```

---

#### `list_export_formats`

List all supported export formats with descriptions.

**Parameters:** None

**Returns:**
```json
{
  "success": true,
  "formats": [
    {
      "format": "markdown",
      "enum": "MARKDOWN",
      "description": "Convert to Markdown format...",
      "use_case": "Best for general-purpose text extraction...",
      "file_extension": ".md"
    }
  ]
}
```

**Example Usage:**
```
What export formats are supported?
```

---

#### `validate_file`

Validate that a file exists and is supported.

**Parameters:**
- `file_path` (required): Path to file to validate

**Returns:**
```json
{
  "success": true,
  "file_path": "/path/to/doc.pdf",
  "file_name": "doc.pdf",
  "exists": true,
  "supported": true,
  "file_type": "document",
  "extension": ".pdf",
  "size_bytes": 2048576,
  "size_mb": 2.0,
  "supported_extensions": {
    "documents": [".pdf", ".docx", ...],
    "images": [".png", ".jpg", ...],
    "audio": [".wav", ".mp3", ...]
  }
}
```

**Example Usage:**
```
Check if myfile.pdf is supported
```

---

#### `estimate_processing_time`

Estimate processing time for documents based on size and operations.

**Parameters:**
- `file_paths` (required): Files to estimate
- `operations` (required): Array of operations (["convert", "chunk", "extract_tables", "extract_images"])
- `hardware` (optional): "auto" | "mps" | "cuda" | "cpu" (default: "auto")

**Returns:**
```json
{
  "success": true,
  "file_count": 10,
  "valid_files": 10,
  "total_size_mb": 50.5,
  "hardware": "mps",
  "operations": ["convert", "chunk"],
  "estimated_seconds": 45.3,
  "estimated_time_human": "45.3 seconds",
  "note": "Estimates are approximate and will vary..."
}
```

**Example Usage:**
```
How long will it take to convert and chunk these 10 PDFs on MPS?
```

---

## Common Workflows

### Workflow 1: Document to RAG Chunks

```
1. validate_file - Check if document is supported
2. convert_document - Convert to markdown
3. chunk_document - Create RAG-ready chunks
4. (Use chunks in your RAG pipeline)
```

### Workflow 2: Batch Document Processing

```
1. detect_hardware - Check available acceleration
2. estimate_processing_time - Plan the job
3. batch_convert_documents - Convert all documents
4. batch processing of chunks
```

### Workflow 3: Audio + Document Multi-Modal

```
1. transcribe_audio - Convert meeting recording to text
2. chunk_text - Chunk the transcription
3. convert_document - Convert meeting slides
4. chunk_document - Chunk the slides
5. Combine both for comprehensive context
```

### Workflow 4: Extract Structured Data

```
1. extract_tables - Get tables from financial report
2. extract_images - Get charts and diagrams
3. Use structured data for analysis
```

---

## Error Handling

All tools return a consistent error format:

```json
{
  "success": false,
  "error": "File not found: /path/to/missing.pdf",
  "file_path": "/path/to/missing.pdf"
}
```

For exceptions, the full traceback is included:

```json
{
  "success": false,
  "error": {
    "type": "ValueError",
    "message": "Invalid model size",
    "traceback": "..."
  }
}
```

---

##Performance Tips

1. **Use Hardware Acceleration**: Run `detect_hardware` first and use the recommended setting
2. **Batch Operations**: Use `batch_convert_documents` and `batch_transcribe_audio` for multiple files
3. **Estimate First**: Use `estimate_processing_time` for large jobs
4. **Choose Right Model Size**: Smaller Whisper models (tiny, base) are much faster
5. **Optimize Chunk Size**: Use `analyze_chunk_distribution` to find optimal parameters

---

## Troubleshooting

### "MCP server not responding"

Check that dependencies are installed:
```bash
uv sync --extra mcp
```

### "Hardware acceleration not working"

Run `detect_hardware` tool to verify available options. MPS requires Apple Silicon, CUDA requires NVIDIA GPU.

### "Out of memory during batch processing"

Reduce `max_workers` parameter in batch operations. Audio transcription is especially memory-intensive.

### "Chunks are too large/small"

Use `analyze_chunk_distribution` to understand your document's characteristics, then adjust `max_tokens` and `overlap` accordingly.

---

## Development

### Adding New Tools

1. Create tool implementation in `src/vertector_data_ingestion/mcp/tools/`
2. Add tool definition to `list_tools()` in `server.py`
3. Add tool handler in `call_tool()` in `server.py`
4. Update documentation

### Testing

```bash
# Run MCP server in development mode
uv run python -m vertector_data_ingestion.mcp.server

# Test with MCP inspector
npx @modelcontextprotocol/inspector uv --directory . run vertector-mcp
```

---

## License

Same as main project license.
