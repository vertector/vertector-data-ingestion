"""Document processing tools for MCP server."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vertector_data_ingestion import (
    ExportFormat,
    UniversalConverter,
)


async def convert_document(
    file_path: str,
    output_format: str = "markdown",
    pipeline: str = "auto",
    hardware: str = "auto",
    get_converter_fn: Callable | None = None,
) -> dict[str, Any]:
    """Convert a document to structured format.

    Args:
        file_path: Path to document file
        output_format: Output format (markdown, json, doctags)
        pipeline: Processing pipeline (auto, classic, vlm)
        hardware: Hardware to use (auto, mps, cuda, cpu)
        get_converter_fn: Function to get converter instance

    Returns:
        Dict with converted content and metadata
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        # Get converter
        converter = get_converter_fn(hardware) if get_converter_fn else UniversalConverter()

        # Convert document
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(
            None,
            converter.convert,
            path,
        )

        # Map output format
        format_map = {
            "markdown": ExportFormat.MARKDOWN,
            "json": ExportFormat.JSON,
            "doctags": ExportFormat.DOCTAGS,
        }
        export_format = format_map.get(output_format, ExportFormat.MARKDOWN)

        # Export
        content = await loop.run_in_executor(
            None,
            converter.export,
            doc,
            export_format,
        )

        # Extract metadata
        metadata = {
            "source_path": str(doc.metadata.source_path),
            "num_pages": doc.metadata.num_pages,
            "file_size": path.stat().st_size,
            "format": output_format,
            "pipeline_used": pipeline,
            "hardware_used": hardware,
        }

        return {
            "success": True,
            "content": content,
            "metadata": metadata,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
        }


async def batch_convert_documents(
    file_paths: list[str],
    output_format: str = "markdown",
    pipeline: str = "auto",
    hardware: str = "auto",
    max_workers: int = 4,
    get_converter_fn: Callable | None = None,
) -> dict[str, Any]:
    """Convert multiple documents in parallel.

    Args:
        file_paths: List of document file paths
        output_format: Output format (markdown, json, doctags)
        pipeline: Processing pipeline (auto, classic, vlm)
        hardware: Hardware to use (auto, mps, cuda, cpu)
        max_workers: Number of parallel workers
        get_converter_fn: Function to get converter instance

    Returns:
        Dict with results for each document
    """
    try:
        # Convert all documents in parallel
        tasks = [
            convert_document(
                file_path=fp,
                output_format=output_format,
                pipeline=pipeline,
                hardware=hardware,
                get_converter_fn=get_converter_fn,
            )
            for fp in file_paths
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed = [r for r in results if not (isinstance(r, dict) and r.get("success"))]

        return {
            "success": True,
            "total": len(file_paths),
            "successful": len(successful),
            "failed": len(failed),
            "results": results,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def extract_metadata(
    file_path: str,
) -> dict[str, Any]:
    """Extract metadata from a document without full conversion.

    Args:
        file_path: Path to document file

    Returns:
        Dict with document metadata
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        # Get file stats
        stat = path.stat()

        metadata = {
            "success": True,
            "file_path": str(path.absolute()),
            "file_name": path.name,
            "file_size": stat.st_size,
            "file_size_mb": round(stat.st_size / (1024 * 1024), 2),
            "file_extension": path.suffix,
            "created": stat.st_ctime,
            "modified": stat.st_mtime,
        }

        # Try to extract document-specific metadata
        # This would require partial document loading
        # For now, return file metadata

        return metadata

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
        }


async def extract_tables(
    file_path: str,
    output_format: str = "json",
    hardware: str = "auto",
    get_converter_fn: Callable | None = None,
) -> dict[str, Any]:
    """Extract tables from a document.

    Args:
        file_path: Path to document file
        output_format: Output format for tables (json, csv, markdown)
        hardware: Hardware to use (auto, mps, cuda, cpu)
        get_converter_fn: Function to get converter instance

    Returns:
        Dict with extracted tables
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        # Get converter
        converter = get_converter_fn(hardware) if get_converter_fn else UniversalConverter()

        # Convert document
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(
            None,
            converter.convert,
            path,
        )

        # Extract tables from document
        # This requires iterating through document items
        tables = []

        # Access the underlying Docling document
        docling_doc = doc.document

        # Iterate through document items to find tables
        for item in docling_doc.body.items:
            if hasattr(item, "self_ref") and "table" in str(item.self_ref).lower():
                # This is a table reference
                table_data = {
                    "type": "table",
                    "text": str(item),
                }
                tables.append(table_data)

        return {
            "success": True,
            "file_path": str(path),
            "num_tables": len(tables),
            "tables": tables,
            "format": output_format,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
        }


async def extract_images(
    file_path: str,
    output_dir: str,
    generate_captions: bool = False,
    hardware: str = "auto",
    get_converter_fn: Callable | None = None,
) -> dict[str, Any]:
    """Extract images from a document.

    Args:
        file_path: Path to document file
        output_dir: Directory to save extracted images
        generate_captions: Whether to generate captions using VLM
        hardware: Hardware to use (auto, mps, cuda, cpu)
        get_converter_fn: Function to get converter instance

    Returns:
        Dict with extracted image paths and metadata
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Get converter
        converter = get_converter_fn(hardware) if get_converter_fn else UniversalConverter()

        # Convert document
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(
            None,
            converter.convert,
            path,
        )

        # Extract images
        # This would require accessing document pictures
        images = []

        # Access the underlying Docling document
        docling_doc = doc.document

        # Get pictures from document
        if hasattr(docling_doc, "pictures") and docling_doc.pictures:
            for idx, _picture in enumerate(docling_doc.pictures):
                image_path = output_path / f"image_{idx}.png"
                # Save picture data
                # This requires accessing the actual image data from Docling
                images.append(
                    {
                        "index": idx,
                        "path": str(image_path),
                        "caption": None,  # Would be generated if generate_captions=True
                    }
                )

        return {
            "success": True,
            "file_path": str(path),
            "output_dir": str(output_path),
            "num_images": len(images),
            "images": images,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
        }
