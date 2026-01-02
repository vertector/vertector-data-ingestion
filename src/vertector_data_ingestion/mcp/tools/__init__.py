"""MCP tool implementations."""

from .audio_tools import (
    batch_transcribe_audio,
    transcribe_audio,
)
from .chunking_tools import (
    analyze_chunk_distribution,
    chunk_document,
    chunk_text,
)
from .document_tools import (
    batch_convert_documents,
    convert_document,
    extract_images,
    extract_metadata,
    extract_tables,
)
from .utility_tools import (
    detect_hardware,
    estimate_processing_time,
    list_export_formats,
    validate_file,
)

__all__ = [
    # Document tools
    "convert_document",
    "batch_convert_documents",
    "extract_metadata",
    "extract_tables",
    "extract_images",
    # Chunking tools
    "chunk_document",
    "chunk_text",
    "analyze_chunk_distribution",
    # Audio tools
    "transcribe_audio",
    "batch_transcribe_audio",
    # Utility tools
    "detect_hardware",
    "list_export_formats",
    "validate_file",
    "estimate_processing_time",
]
