"""Utility tools for MCP server."""

from pathlib import Path
from typing import Any

from vertector_data_ingestion.core.hardware_detector import HardwareDetector


async def detect_hardware() -> dict[str, Any]:
    """Detect available hardware acceleration options.

    Returns:
        Dict with available hardware and recommendations
    """
    try:
        # Get hardware configuration
        hw_config = HardwareDetector.detect()
        hw_info = HardwareDetector.get_device_info()

        # Check what's available
        has_mps = hw_config.device_type.value == "mps"
        has_cuda = hw_config.device_type.value == "cuda"

        hardware_info = {
            "success": True,
            "available": {
                "mps": has_mps,
                "cuda": has_cuda,
                "cpu": True,  # Always available
            },
            "detected_device": hw_config.device_type.value,
            "device_info": hw_info,
        }

        # Add recommendations
        if has_mps:
            hardware_info["recommended"] = "mps"
            hardware_info["recommendation_reason"] = (
                f"Apple Silicon ({hw_info.get('chip', 'unknown')}) MPS detected - optimal for VLM and audio"
            )
        elif has_cuda:
            gpu_name = hw_info.get("gpu_name", "Unknown GPU")
            gpu_memory = hw_info.get("gpu_memory_gb", 0)
            hardware_info["recommended"] = "cuda"
            hardware_info["recommendation_reason"] = (
                f"NVIDIA GPU detected: {gpu_name} ({gpu_memory:.1f} GB)"
            )
            hardware_info["cuda_devices"] = 1
        else:
            hardware_info["recommended"] = "cpu"
            hardware_info["recommendation_reason"] = "No GPU acceleration detected - using CPU"
            hardware_info["cpu_count"] = hw_info.get("cpu_count", "unknown")

        return hardware_info

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def list_export_formats() -> dict[str, Any]:
    """List all supported export formats with descriptions.

    Returns:
        Dict with export formats and their descriptions
    """
    try:
        formats = [
            {
                "format": "markdown",
                "enum": "MARKDOWN",
                "description": "Convert to Markdown format with preserved structure, tables, and formatting",
                "use_case": "Best for general-purpose text extraction and human readability",
                "file_extension": ".md",
            },
            {
                "format": "json",
                "enum": "JSON",
                "description": "Export as structured JSON with full document hierarchy and metadata",
                "use_case": "Best for programmatic access and maintaining document structure",
                "file_extension": ".json",
            },
            {
                "format": "doctags",
                "enum": "DOCTAGS",
                "description": "Export with XML-style tags marking document structure (headings, lists, etc.)",
                "use_case": "Best for semantic analysis and structure-aware processing",
                "file_extension": ".txt",
            },
        ]

        return {
            "success": True,
            "formats": formats,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def validate_file(
    file_path: str,
) -> dict[str, Any]:
    """Validate that a file exists and is supported.

    Args:
        file_path: Path to file to validate

    Returns:
        Dict with file validation info
    """
    try:
        path = Path(file_path)

        # Supported extensions
        supported_documents = {
            ".pdf",
            ".docx",
            ".pptx",
            ".xlsx",
            ".doc",
            ".ppt",
            ".xls",  # Legacy formats
            ".html",
            ".htm",
        }
        supported_images = {
            ".png",
            ".jpg",
            ".jpeg",
            ".bmp",
            ".tiff",
            ".tif",
        }
        supported_audio = {
            ".wav",
            ".mp3",
            ".m4a",
            ".flac",
            ".ogg",
        }

        supported_documents | supported_images | supported_audio

        if not path.exists():
            return {
                "success": False,
                "error": "File does not exist",
                "file_path": str(path),
            }

        # Get file info
        stat = path.stat()
        extension = path.suffix.lower()

        # Determine file type
        if extension in supported_documents:
            file_type = "document"
            supported = True
        elif extension in supported_images:
            file_type = "image"
            supported = True
        elif extension in supported_audio:
            file_type = "audio"
            supported = True
        else:
            file_type = "unknown"
            supported = False

        return {
            "success": True,
            "file_path": str(path.absolute()),
            "file_name": path.name,
            "exists": True,
            "supported": supported,
            "file_type": file_type,
            "extension": extension,
            "size_bytes": stat.st_size,
            "size_mb": round(stat.st_size / (1024 * 1024), 2),
            "supported_extensions": {
                "documents": list(supported_documents),
                "images": list(supported_images),
                "audio": list(supported_audio),
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
        }


async def estimate_processing_time(
    file_paths: list[str],
    operations: list[str],
    hardware: str = "auto",
) -> dict[str, Any]:
    """Estimate processing time for documents.

    Args:
        file_paths: Files to estimate processing time for
        operations: Operations to perform (convert, chunk, extract_tables, extract_images)
        hardware: Hardware to use (auto, mps, cuda, cpu)

    Returns:
        Dict with time estimates and resource requirements
    """
    try:
        # Detect hardware if auto
        if hardware == "auto":
            hw_config = HardwareDetector.detect()
            hardware = hw_config.device_type.value

        # Calculate total file size
        total_size_mb = 0
        file_count = len(file_paths)
        valid_files = 0

        for fp in file_paths:
            path = Path(fp)
            if path.exists():
                total_size_mb += path.stat().st_size / (1024 * 1024)
                valid_files += 1

        # Rough estimates (in seconds per MB)
        # These are ballpark figures and will vary significantly
        time_per_mb = {
            "mps": {
                "convert": 0.5,  # VLM on MPS
                "chunk": 0.1,
                "extract_tables": 0.3,
                "extract_images": 0.2,
            },
            "cuda": {
                "convert": 0.4,
                "chunk": 0.1,
                "extract_tables": 0.25,
                "extract_images": 0.15,
            },
            "cpu": {
                "convert": 2.0,  # Much slower without acceleration
                "chunk": 0.2,
                "extract_tables": 1.0,
                "extract_images": 0.8,
            },
        }

        # Calculate estimated time
        estimated_seconds = 0
        for operation in operations:
            if operation in time_per_mb[hardware]:
                estimated_seconds += total_size_mb * time_per_mb[hardware][operation]

        # Add overhead for batch processing
        overhead_seconds = file_count * 0.5  # ~0.5s per file for I/O

        total_estimated_seconds = estimated_seconds + overhead_seconds

        # Format as human-readable
        if total_estimated_seconds < 60:
            time_str = f"{total_estimated_seconds:.1f} seconds"
        elif total_estimated_seconds < 3600:
            time_str = f"{total_estimated_seconds / 60:.1f} minutes"
        else:
            time_str = f"{total_estimated_seconds / 3600:.1f} hours"

        return {
            "success": True,
            "file_count": file_count,
            "valid_files": valid_files,
            "total_size_mb": round(total_size_mb, 2),
            "hardware": hardware,
            "operations": operations,
            "estimated_seconds": round(total_estimated_seconds, 1),
            "estimated_time_human": time_str,
            "note": "Estimates are approximate and will vary based on document complexity and system load",
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }
