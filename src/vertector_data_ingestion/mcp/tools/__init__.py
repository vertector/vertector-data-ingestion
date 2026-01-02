"""MCP tool implementations."""

from .document_tools import (
    convert_document,
    batch_convert_documents,
    extract_metadata,
    extract_tables,
    extract_images,
)
from .chunking_tools import (
    chunk_document,
    chunk_text,
    analyze_chunk_distribution,
)
from .audio_tools import (
    transcribe_audio,
    batch_transcribe_audio,
)
from .utility_tools import (
    detect_hardware,
    list_export_formats,
    validate_file,
    estimate_processing_time,
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
