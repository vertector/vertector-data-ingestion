"""Example of using Vertector MCP server programmatically.

This example shows how to test MCP tools locally without running
the full server.
"""

import asyncio
import json
from pathlib import Path

from vertector_data_ingestion.mcp.tools import (
    convert_document,
    chunk_document,
    transcribe_audio,
    detect_hardware,
    analyze_chunk_distribution,
)


async def example_document_conversion():
    """Example: Convert a PDF to markdown."""
    print("\n" + "=" * 60)
    print("Example 1: Document Conversion")
    print("=" * 60)

    result = await convert_document(
        file_path="test_documents/arxiv_sample.pdf",
        output_format="markdown",
        hardware="auto",
    )

    if result["success"]:
        print(f"✓ Converted {result['metadata']['num_pages']} pages")
        print(f"Content preview:\n{result['content'][:200]}...")
    else:
        print(f"✗ Error: {result['error']}")


async def example_chunking():
    """Example: Create RAG chunks from a document."""
    print("\n" + "=" * 60)
    print("Example 2: Document Chunking")
    print("=" * 60)

    result = await chunk_document(
        file_path="test_documents/arxiv_sample.pdf",
        max_tokens=512,
        overlap=128,
        tokenizer="Qwen/Qwen3-Embedding-0.6B",
        include_metadata=True,
    )

    if result["success"]:
        print(f"✓ Created {result['total_chunks']} chunks")
        print(f"Statistics:")
        for key, value in result["statistics"].items():
            print(f"  {key}: {value}")

        # Show first chunk
        if result["chunks"]:
            chunk = result["chunks"][0]
            print(f"\nFirst chunk preview:")
            print(f"  ID: {chunk['chunk_id']}")
            print(f"  Text: {chunk['text'][:100]}...")
            if "metadata" in chunk:
                print(f"  Page: {chunk['metadata'].get('page_number')}")
    else:
        print(f"✗ Error: {result['error']}")


async def example_audio_transcription():
    """Example: Transcribe an audio file."""
    print("\n" + "=" * 60)
    print("Example 3: Audio Transcription")
    print("=" * 60)

    audio_path = "test_documents/harvard.wav"

    if Path(audio_path).exists():
        result = await transcribe_audio(
            file_path=audio_path,
            model_size="base",
            language="auto",
            include_timestamps=True,
            output_format="text",
        )

        if result["success"]:
            print(f"✓ Transcribed {result['metadata']['duration']:.1f}s audio")
            print(f"Language: {result['metadata']['language']}")
            print(f"Transcription:\n{result['transcription'][:200]}...")
        else:
            print(f"✗ Error: {result['error']}")
    else:
        print(f"⚠ Audio file not found: {audio_path}")


async def example_hardware_detection():
    """Example: Detect available hardware."""
    print("\n" + "=" * 60)
    print("Example 4: Hardware Detection")
    print("=" * 60)

    result = await detect_hardware()

    if result["success"]:
        print("Available hardware:")
        for hw, available in result["available"].items():
            status = "✓" if available else "✗"
            print(f"  {status} {hw.upper()}")

        print(f"\nRecommended: {result['recommended'].upper()}")
        print(f"Reason: {result['recommendation_reason']}")
    else:
        print(f"✗ Error: {result['error']}")


async def example_chunk_analysis():
    """Example: Analyze chunk distribution."""
    print("\n" + "=" * 60)
    print("Example 5: Chunk Distribution Analysis")
    print("=" * 60)

    result = await analyze_chunk_distribution(
        file_path="test_documents/arxiv_sample.pdf",
        max_tokens=512,
        overlap=128,
    )

    if result["success"]:
        print(f"✓ Analyzed {result['total_chunks']} chunks")
        print(f"\nDistribution statistics:")
        for key, value in result["distribution"].items():
            print(f"  {key}: {value:.1f}")

        print(f"\nHistogram:")
        for range_info in result["histogram"]["ranges"]:
            bar = "█" * (range_info["count"] // 2)
            print(f"  {range_info['range']:>10}: {bar} ({range_info['count']})")
    else:
        print(f"✗ Error: {result['error']}")


async def main():
    """Run all examples."""
    print("\n" + "=" * 60)
    print("Vertector MCP Server Examples")
    print("=" * 60)

    # Run examples
    await example_hardware_detection()
    await example_document_conversion()
    await example_chunking()
    await example_chunk_analysis()
    await example_audio_transcription()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
