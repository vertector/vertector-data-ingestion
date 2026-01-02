"""MCP Server for Vertector Data Ingestion.

Provides tools for document parsing, chunking, and metadata enrichment
via the Model Context Protocol (MCP).
"""

import asyncio
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

from vertector_data_ingestion import (
    CloudCpuConfig,
    CloudGpuConfig,
    HybridChunker,
    LocalMpsConfig,
    UniversalConverter,
    setup_logging,
)
from vertector_data_ingestion.core.hardware_detector import HardwareDetector
from vertector_data_ingestion.models.config import (
    ChunkingConfig,
)

from .tools.audio_tools import (
    batch_transcribe_audio,
    transcribe_audio,
)
from .tools.chunking_tools import (
    analyze_chunk_distribution,
    chunk_document,
    chunk_text,
)
from .tools.document_tools import (
    batch_convert_documents,
    convert_document,
    extract_images,
    extract_metadata,
    extract_tables,
)
from .tools.utility_tools import (
    detect_hardware,
    estimate_processing_time,
    list_export_formats,
    validate_file,
)

# Setup logging
setup_logging(log_level="INFO")

# Create MCP server
app = Server("vertector-mcp")

# Global state for converters (cached to avoid re-initialization)
_converters: dict[str, UniversalConverter] = {}
_chunkers: dict[str, HybridChunker] = {}


def get_converter(hardware: str = "auto") -> UniversalConverter:
    """Get or create a converter instance with caching."""
    if hardware in _converters:
        return _converters[hardware]

    if hardware == "auto":
        hw_config = HardwareDetector.detect()
        if hw_config.device_type.value == "mps":
            config = LocalMpsConfig()
        elif hw_config.device_type.value == "cuda":
            config = CloudGpuConfig()
        else:
            config = CloudCpuConfig()
    elif hardware == "mps":
        config = LocalMpsConfig()
    elif hardware == "cuda":
        config = CloudGpuConfig()
    else:
        config = CloudCpuConfig()

    converter = UniversalConverter(config)
    _converters[hardware] = converter
    return converter


def get_chunker(
    tokenizer: str = "Qwen/Qwen3-Embedding-0.6B",
    max_tokens: int = 512,
) -> HybridChunker:
    """Get or create a chunker instance with caching."""
    cache_key = f"{tokenizer}_{max_tokens}"

    if cache_key in _chunkers:
        return _chunkers[cache_key]

    config = ChunkingConfig(
        tokenizer=tokenizer,
        max_tokens=max_tokens,
    )
    chunker = HybridChunker(config=config)
    _chunkers[cache_key] = chunker
    return chunker


# ============================================================================
# Tool Definitions
# ============================================================================


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available MCP tools."""
    return [
        # Document Processing Tools
        Tool(
            name="convert_document",
            description=(
                "Convert a document (PDF, DOCX, PPTX, XLSX, images) to structured format. "
                "Supports automatic pipeline selection (classic or VLM) and hardware acceleration. "
                "Returns converted content with metadata including page count, tables, and structure."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the document file (absolute or relative)",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["markdown", "json", "doctags"],
                        "default": "markdown",
                        "description": "Output format for the converted document",
                    },
                    "pipeline": {
                        "type": "string",
                        "enum": ["auto", "classic", "vlm"],
                        "default": "auto",
                        "description": "Processing pipeline to use (auto-selects by default)",
                    },
                    "hardware": {
                        "type": "string",
                        "enum": ["auto", "mps", "cuda", "cpu"],
                        "default": "auto",
                        "description": "Hardware acceleration to use",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="batch_convert_documents",
            description=(
                "Convert multiple documents in parallel. Useful for processing entire folders. "
                "Returns results for each document with success/failure status."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of document file paths to convert",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["markdown", "json", "doctags"],
                        "default": "markdown",
                    },
                    "pipeline": {
                        "type": "string",
                        "enum": ["auto", "classic", "vlm"],
                        "default": "auto",
                    },
                    "hardware": {
                        "type": "string",
                        "enum": ["auto", "mps", "cuda", "cpu"],
                        "default": "auto",
                    },
                    "max_workers": {
                        "type": "integer",
                        "default": 4,
                        "description": "Number of parallel workers for processing",
                    },
                },
                "required": ["file_paths"],
            },
        ),
        Tool(
            name="extract_metadata",
            description=(
                "Extract metadata from a document without full conversion. "
                "Returns title, author, page count, creation date, file size, etc. "
                "Fast operation useful for cataloging documents."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the document file",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="extract_tables",
            description=(
                "Extract all tables from a document with structure preserved. "
                "Returns tables in JSON, CSV, or Markdown format. "
                "Useful for extracting data from reports, spreadsheets, etc."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the document file",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["json", "csv", "markdown"],
                        "default": "json",
                        "description": "Format for extracted tables",
                    },
                    "hardware": {
                        "type": "string",
                        "enum": ["auto", "mps", "cuda", "cpu"],
                        "default": "auto",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="extract_images",
            description=(
                "Extract all images from a document (embedded images in PDFs, etc.). "
                "Optionally generate captions using VLM. "
                "Returns image paths and metadata (dimensions, format, captions)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the document file",
                    },
                    "output_dir": {
                        "type": "string",
                        "description": "Directory to save extracted images",
                    },
                    "generate_captions": {
                        "type": "boolean",
                        "default": False,
                        "description": "Use VLM to generate captions for images",
                    },
                    "hardware": {
                        "type": "string",
                        "enum": ["auto", "mps", "cuda", "cpu"],
                        "default": "auto",
                    },
                },
                "required": ["file_path", "output_dir"],
            },
        ),
        # Chunking Tools
        Tool(
            name="chunk_document",
            description=(
                "Create semantic chunks from a document for RAG applications. "
                "Uses hybrid chunking strategy with token-aware splitting. "
                "Returns chunks with text, metadata, bounding boxes, and hierarchy info."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the document file to chunk",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "default": 512,
                        "description": "Maximum tokens per chunk",
                    },
                    "tokenizer": {
                        "type": "string",
                        "default": "Qwen/Qwen3-Embedding-0.6B",
                        "description": "HuggingFace tokenizer model to use",
                    },
                    "include_metadata": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include rich metadata (bounding boxes, hierarchy, etc.)",
                    },
                    "hardware": {
                        "type": "string",
                        "enum": ["auto", "mps", "cuda", "cpu"],
                        "default": "auto",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="chunk_text",
            description=(
                "Chunk raw text directly without file input. "
                "Useful for chunking text from APIs, scraped content, etc. "
                "Uses same hybrid chunking strategy as chunk_document."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Raw text to chunk",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "default": 512,
                    },
                    "tokenizer": {
                        "type": "string",
                        "default": "Qwen/Qwen3-Embedding-0.6B",
                    },
                    "doc_id": {
                        "type": "string",
                        "description": "Optional document ID for metadata",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="analyze_chunk_distribution",
            description=(
                "Analyze chunk size distribution for a document. "
                "Returns statistics: min/max/mean chunk sizes, token count distribution. "
                "Helps optimize chunking parameters before processing large document sets."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to the document file",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "default": 512,
                    },
                    "tokenizer": {
                        "type": "string",
                        "default": "Qwen/Qwen3-Embedding-0.6B",
                    },
                    "hardware": {
                        "type": "string",
                        "enum": ["auto", "mps", "cuda", "cpu"],
                        "default": "auto",
                    },
                },
                "required": ["file_path"],
            },
        ),
        # Audio Tools
        Tool(
            name="transcribe_audio",
            description=(
                "Transcribe audio file to text using Whisper. "
                "Supports multiple model sizes and backends (MLX for Apple Silicon, CUDA, CPU). "
                "Returns transcription with timestamps, detected language, and confidence scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to audio file",
                    },
                    "model_size": {
                        "type": "string",
                        "enum": ["tiny", "base", "small", "medium", "large"],
                        "default": "base",
                        "description": "Whisper model size (larger = more accurate but slower)",
                    },
                    "language": {
                        "type": "string",
                        "default": "auto",
                        "description": "Language code (e.g., 'en', 'es') or 'auto' for detection",
                    },
                    "include_timestamps": {
                        "type": "boolean",
                        "default": True,
                        "description": "Include word/segment timestamps",
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["text", "srt", "json"],
                        "default": "text",
                        "description": "Output format for transcription",
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["auto", "mlx", "standard"],
                        "default": "auto",
                        "description": "Whisper backend (auto-detects best option)",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="batch_transcribe_audio",
            description=(
                "Transcribe multiple audio files in parallel. "
                "Returns array of transcription results with success/failure status."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Array of audio file paths",
                    },
                    "model_size": {
                        "type": "string",
                        "enum": ["tiny", "base", "small", "medium", "large"],
                        "default": "base",
                    },
                    "language": {
                        "type": "string",
                        "default": "auto",
                    },
                    "include_timestamps": {
                        "type": "boolean",
                        "default": True,
                    },
                    "output_format": {
                        "type": "string",
                        "enum": ["text", "srt", "json"],
                        "default": "text",
                    },
                    "backend": {
                        "type": "string",
                        "enum": ["auto", "mlx", "standard"],
                        "default": "auto",
                    },
                    "max_workers": {
                        "type": "integer",
                        "default": 2,
                        "description": "Parallel workers (audio processing is memory-intensive)",
                    },
                },
                "required": ["file_paths"],
            },
        ),
        # Utility Tools
        Tool(
            name="detect_hardware",
            description=(
                "Detect available hardware acceleration options. "
                "Returns available backends (MPS, CUDA, CPU) with recommendations "
                "for optimal performance based on your system."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="list_export_formats",
            description=(
                "List all supported export formats with descriptions. "
                "Helps choose the right format for your use case."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
            },
        ),
        Tool(
            name="validate_file",
            description=(
                "Validate that a file exists and is supported. "
                "Returns file info: size, type, format, and whether it's supported. "
                "Useful for checking files before processing."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "Path to file to validate",
                    },
                },
                "required": ["file_path"],
            },
        ),
        Tool(
            name="estimate_processing_time",
            description=(
                "Estimate processing time for documents based on size and operations. "
                "Returns estimated time and resource requirements. "
                "Helps plan batch processing jobs."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "file_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files to estimate processing time for",
                    },
                    "operations": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": ["convert", "chunk", "extract_tables", "extract_images"],
                        },
                        "description": "Operations to perform",
                    },
                    "hardware": {
                        "type": "string",
                        "enum": ["auto", "mps", "cuda", "cpu"],
                        "default": "auto",
                    },
                },
                "required": ["file_paths", "operations"],
            },
        ),
    ]


# ============================================================================
# Tool Handlers
# ============================================================================


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls from MCP clients."""
    try:
        if name == "convert_document":
            result = await convert_document(
                file_path=arguments["file_path"],
                output_format=arguments.get("output_format", "markdown"),
                pipeline=arguments.get("pipeline", "auto"),
                hardware=arguments.get("hardware", "auto"),
                get_converter_fn=get_converter,
            )

        elif name == "batch_convert_documents":
            result = await batch_convert_documents(
                file_paths=arguments["file_paths"],
                output_format=arguments.get("output_format", "markdown"),
                pipeline=arguments.get("pipeline", "auto"),
                hardware=arguments.get("hardware", "auto"),
                max_workers=arguments.get("max_workers", 4),
                get_converter_fn=get_converter,
            )

        elif name == "extract_metadata":
            result = await extract_metadata(
                file_path=arguments["file_path"],
            )

        elif name == "extract_tables":
            result = await extract_tables(
                file_path=arguments["file_path"],
                output_format=arguments.get("output_format", "json"),
                hardware=arguments.get("hardware", "auto"),
                get_converter_fn=get_converter,
            )

        elif name == "extract_images":
            result = await extract_images(
                file_path=arguments["file_path"],
                output_dir=arguments["output_dir"],
                generate_captions=arguments.get("generate_captions", False),
                hardware=arguments.get("hardware", "auto"),
                get_converter_fn=get_converter,
            )

        elif name == "chunk_document":
            result = await chunk_document(
                file_path=arguments["file_path"],
                max_tokens=arguments.get("max_tokens", 512),
                tokenizer=arguments.get("tokenizer", "Qwen/Qwen3-Embedding-0.6B"),
                include_metadata=arguments.get("include_metadata", True),
                hardware=arguments.get("hardware", "auto"),
                get_converter_fn=get_converter,
                get_chunker_fn=get_chunker,
            )

        elif name == "chunk_text":
            result = await chunk_text(
                text=arguments["text"],
                max_tokens=arguments.get("max_tokens", 512),
                tokenizer=arguments.get("tokenizer", "Qwen/Qwen3-Embedding-0.6B"),
                doc_id=arguments.get("doc_id"),
                get_chunker_fn=get_chunker,
            )

        elif name == "analyze_chunk_distribution":
            result = await analyze_chunk_distribution(
                file_path=arguments["file_path"],
                max_tokens=arguments.get("max_tokens", 512),
                tokenizer=arguments.get("tokenizer", "Qwen/Qwen3-Embedding-0.6B"),
                hardware=arguments.get("hardware", "auto"),
                get_converter_fn=get_converter,
                get_chunker_fn=get_chunker,
            )

        elif name == "transcribe_audio":
            result = await transcribe_audio(
                file_path=arguments["file_path"],
                model_size=arguments.get("model_size", "base"),
                language=arguments.get("language", "auto"),
                include_timestamps=arguments.get("include_timestamps", True),
                output_format=arguments.get("output_format", "text"),
                backend=arguments.get("backend", "auto"),
            )

        elif name == "batch_transcribe_audio":
            result = await batch_transcribe_audio(
                file_paths=arguments["file_paths"],
                model_size=arguments.get("model_size", "base"),
                language=arguments.get("language", "auto"),
                include_timestamps=arguments.get("include_timestamps", True),
                output_format=arguments.get("output_format", "text"),
                backend=arguments.get("backend", "auto"),
                max_workers=arguments.get("max_workers", 2),
            )

        elif name == "detect_hardware":
            result = await detect_hardware()

        elif name == "list_export_formats":
            result = await list_export_formats()

        elif name == "validate_file":
            result = await validate_file(
                file_path=arguments["file_path"],
            )

        elif name == "estimate_processing_time":
            result = await estimate_processing_time(
                file_paths=arguments["file_paths"],
                operations=arguments["operations"],
                hardware=arguments.get("hardware", "auto"),
            )

        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]

        # Return result as TextContent
        import json

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        import traceback

        error_result = {
            "success": False,
            "error": {
                "type": type(e).__name__,
                "message": str(e),
                "traceback": traceback.format_exc(),
            },
        }
        import json

        return [TextContent(type="text", text=json.dumps(error_result, indent=2))]


async def async_main():
    """Run the MCP server (async entry point)."""
    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def main():
    """Run the MCP server (sync entry point for CLI)."""
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
