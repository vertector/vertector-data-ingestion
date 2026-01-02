"""Chunking tools for MCP server."""

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from vertector_data_ingestion import HybridChunker


async def chunk_document(
    file_path: str,
    max_tokens: int = 512,
    tokenizer: str = "Qwen/Qwen3-Embedding-0.6B",
    include_metadata: bool = True,
    hardware: str = "auto",
    get_converter_fn: Callable | None = None,
    get_chunker_fn: Callable | None = None,
) -> dict[str, Any]:
    """Create semantic chunks from a document.

    Args:
        file_path: Path to document file
        max_tokens: Maximum tokens per chunk
        tokenizer: HuggingFace tokenizer model
        include_metadata: Include rich metadata in chunks
        hardware: Hardware to use for document conversion
        get_converter_fn: Function to get converter instance
        get_chunker_fn: Function to get chunker instance

    Returns:
        Dict with chunks and statistics
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        # Get converter and chunker
        if get_converter_fn:
            converter = get_converter_fn(hardware)
        else:
            from vertector_data_ingestion import UniversalConverter

            converter = UniversalConverter()

        if get_chunker_fn:
            chunker = get_chunker_fn(tokenizer, max_tokens)
        else:
            from vertector_data_ingestion.models.config import ChunkingConfig

            config = ChunkingConfig(
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
            chunker = HybridChunker(config=config)

        # Convert document
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(
            None,
            converter.convert,
            path,
        )

        # Chunk document
        chunking_result = await loop.run_in_executor(
            None,
            chunker.chunk_document,
            doc,
        )

        # Format chunks for output
        chunks_data = []
        for chunk in chunking_result.chunks:
            chunk_dict = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "chunk_index": chunk.chunk_index,
                "total_chunks": chunk.total_chunks,
            }

            if include_metadata:
                chunk_dict["metadata"] = {
                    "doc_id": chunk.doc_id,
                    "source_path": str(chunk.source_path) if chunk.source_path else None,
                    "page_number": chunk.page_number,
                    "section_hierarchy": chunk.section_hierarchy,
                }

                # Add bounding box if available
                if chunk.bbox:
                    chunk_dict["metadata"]["bbox"] = {
                        "left": chunk.bbox.l,
                        "top": chunk.bbox.t,
                        "right": chunk.bbox.r,
                        "bottom": chunk.bbox.b,
                    }

            chunks_data.append(chunk_dict)

        # Calculate statistics
        chunk_sizes = [len(chunk.text) for chunk in chunking_result.chunks]
        [chunk.chunk_index for chunk in chunking_result.chunks]

        return {
            "success": True,
            "file_path": str(path),
            "total_chunks": chunking_result.total_chunks,
            "chunks": chunks_data,
            "statistics": {
                "total_chunks": len(chunks_data),
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
            },
            "config": {
                "max_tokens": max_tokens,
                "tokenizer": tokenizer,
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
        }


async def chunk_text(
    text: str,
    max_tokens: int = 512,
    tokenizer: str = "Qwen/Qwen3-Embedding-0.6B",
    doc_id: str | None = None,
    get_chunker_fn: Callable | None = None,
) -> dict[str, Any]:
    """Chunk raw text directly.

    Args:
        text: Raw text to chunk
        max_tokens: Maximum tokens per chunk
        tokenizer: HuggingFace tokenizer model
        doc_id: Optional document ID for metadata
        get_chunker_fn: Function to get chunker instance

    Returns:
        Dict with chunks and statistics
    """
    try:
        # Get chunker
        if get_chunker_fn:
            get_chunker_fn(tokenizer, max_tokens)
        else:
            from vertector_data_ingestion.models.config import ChunkingConfig

            config = ChunkingConfig(
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
            HybridChunker(config=config)

        # Create a minimal document wrapper for chunking
        # We'll use the chunker's internal text chunking method
        from transformers import AutoTokenizer

        # Load tokenizer
        tokenizer_instance = AutoTokenizer.from_pretrained(tokenizer)

        # Tokenize text
        tokens = tokenizer_instance.encode(text)

        # Split into chunks (no overlap, sequential chunking)
        chunks_data = []
        chunk_idx = 0

        for i in range(0, len(tokens), max_tokens):
            chunk_tokens = tokens[i : i + max_tokens]
            chunk_text = tokenizer_instance.decode(chunk_tokens, skip_special_tokens=True)

            chunks_data.append(
                {
                    "chunk_id": f"{doc_id or 'text'}_{chunk_idx}",
                    "text": chunk_text,
                    "chunk_index": chunk_idx,
                    "token_count": len(chunk_tokens),
                }
            )

            chunk_idx += 1

        # Calculate statistics
        chunk_sizes = [len(chunk["text"]) for chunk in chunks_data]
        token_counts = [chunk["token_count"] for chunk in chunks_data]

        return {
            "success": True,
            "total_chunks": len(chunks_data),
            "chunks": chunks_data,
            "statistics": {
                "total_chunks": len(chunks_data),
                "total_tokens": len(tokens),
                "min_chunk_size": min(chunk_sizes) if chunk_sizes else 0,
                "max_chunk_size": max(chunk_sizes) if chunk_sizes else 0,
                "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                "min_token_count": min(token_counts) if token_counts else 0,
                "max_token_count": max(token_counts) if token_counts else 0,
                "avg_token_count": sum(token_counts) / len(token_counts) if token_counts else 0,
            },
            "config": {
                "max_tokens": max_tokens,
                "tokenizer": tokenizer,
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


async def analyze_chunk_distribution(
    file_path: str,
    max_tokens: int = 512,
    tokenizer: str = "Qwen/Qwen3-Embedding-0.6B",
    hardware: str = "auto",
    get_converter_fn: Callable | None = None,
    get_chunker_fn: Callable | None = None,
) -> dict[str, Any]:
    """Analyze chunk size distribution for a document.

    Args:
        file_path: Path to document file
        max_tokens: Maximum tokens per chunk
        tokenizer: HuggingFace tokenizer model
        hardware: Hardware to use for document conversion
        get_converter_fn: Function to get converter instance
        get_chunker_fn: Function to get chunker instance

    Returns:
        Dict with chunk distribution statistics
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        # Get converter and chunker
        if get_converter_fn:
            converter = get_converter_fn(hardware)
        else:
            from vertector_data_ingestion import UniversalConverter

            converter = UniversalConverter()

        if get_chunker_fn:
            chunker = get_chunker_fn(tokenizer, max_tokens)
        else:
            from vertector_data_ingestion.models.config import ChunkingConfig

            config = ChunkingConfig(
                tokenizer=tokenizer,
                max_tokens=max_tokens,
            )
            chunker = HybridChunker(config=config)

        # Convert document
        loop = asyncio.get_event_loop()
        doc = await loop.run_in_executor(
            None,
            converter.convert,
            path,
        )

        # Chunk document
        chunking_result = await loop.run_in_executor(
            None,
            chunker.chunk_document,
            doc,
        )

        # Analyze distribution
        chunk_sizes = [len(chunk.text) for chunk in chunking_result.chunks]
        chunk_sizes_sorted = sorted(chunk_sizes)

        # Calculate percentiles
        def percentile(data, p):
            if not data:
                return 0
            k = (len(data) - 1) * p / 100
            f = int(k)
            c = f + 1 if f + 1 < len(data) else f
            if f == c:
                return data[f]
            return data[f] * (c - k) + data[c] * (k - f)

        return {
            "success": True,
            "file_path": str(path),
            "total_chunks": len(chunk_sizes),
            "distribution": {
                "min": min(chunk_sizes) if chunk_sizes else 0,
                "max": max(chunk_sizes) if chunk_sizes else 0,
                "mean": sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0,
                "median": percentile(chunk_sizes_sorted, 50),
                "p25": percentile(chunk_sizes_sorted, 25),
                "p75": percentile(chunk_sizes_sorted, 75),
                "p90": percentile(chunk_sizes_sorted, 90),
                "p99": percentile(chunk_sizes_sorted, 99),
            },
            "histogram": {
                "ranges": [
                    {"range": "0-100", "count": sum(1 for s in chunk_sizes if s <= 100)},
                    {"range": "101-500", "count": sum(1 for s in chunk_sizes if 100 < s <= 500)},
                    {"range": "501-1000", "count": sum(1 for s in chunk_sizes if 500 < s <= 1000)},
                    {
                        "range": "1001-2000",
                        "count": sum(1 for s in chunk_sizes if 1000 < s <= 2000),
                    },
                    {"range": "2001+", "count": sum(1 for s in chunk_sizes if s > 2000)},
                ],
            },
            "config": {
                "max_tokens": max_tokens,
                "tokenizer": tokenizer,
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
        }
