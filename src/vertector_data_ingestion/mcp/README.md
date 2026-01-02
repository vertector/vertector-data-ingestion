# Vertector Data Ingestion MCP Server

MCP server providing document processing, chunking, and audio transcription tools.

## Quick Start

### Installation

```bash
# Install with MCP dependencies
uv sync --extra mcp
```

### Running the Server

```bash
# Via CLI entry point
vertector-data-ingestion-mcp

# Or via Python module
python -m vertector_data_ingestion.mcp.server
```

### Claude Desktop Configuration

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
      ],
      "env": {
        "VERTECTOR_LOG_LEVEL": "INFO"
      }
    }
  }
}
```

## Available Tools

### Document Processing (5 tools)
- `convert_document` - Convert documents to markdown/json/doctags
- `batch_convert_documents` - Batch conversion
- `extract_metadata` - Fast metadata extraction
- `extract_tables` - Extract tables with structure
- `extract_images` - Extract and caption images

### Chunking (3 tools)
- `chunk_document` - Create RAG-ready chunks
- `chunk_text` - Chunk raw text
- `analyze_chunk_distribution` - Analyze chunk statistics

### Audio (2 tools)
- `transcribe_audio` - Whisper transcription
- `batch_transcribe_audio` - Batch audio processing

### Utilities (4 tools)
- `detect_hardware` - Check available acceleration
- `list_export_formats` - Show supported formats
- `validate_file` - Check file support
- `estimate_processing_time` - Estimate job duration

## Documentation

See [docs/mcp-server.md](../../../docs/mcp-server.md) for full documentation.

## Troubleshooting

### Server won't start

Check logs:
```bash
tail -f "$HOME/Library/Logs/Claude/mcp-server-vertector-data-ingestion-mcp.log"
```

### Import errors

Ensure MCP dependencies are installed:
```bash
uv sync --extra mcp
```

### Server disconnects immediately

Check that the `main()` function is synchronous (not async). The entry point in pyproject.toml should call a synchronous wrapper.
