"""Neo4j SimpleKGPipeline integration components.

This module provides production-ready adapters for integrating Vertector Data Ingestion
with Neo4j's SimpleKGPipeline for multimodal knowledge graph construction.

Installation:
    uv pip install vertector-data-ingestion[neo4j]

Example:
    >>> from vertector_data_ingestion.integrations.neo4j import (
    ...     VertectorDataLoader,
    ...     VertectorAudioLoader,
    ...     VertectorTextSplitter,
    ...     MultimodalLoader,
    ... )
    >>> from neo4j_graphrag.experimental.pipeline import SimpleKGPipeline
    >>>
    >>> # Use multimodal loader
    >>> loader = MultimodalLoader()
    >>> splitter = VertectorTextSplitter(chunk_size=512)
    >>>
    >>> # Create pipeline
    >>> pipeline = SimpleKGPipeline(
    ...     llm=llm,
    ...     driver=driver,
    ...     embedder=embedder,
    ...     pdf_loader=loader,
    ...     text_splitter=splitter,
    ... )
    >>>
    >>> # Process documents and audio
    >>> await pipeline.run_async(file_path="document.pdf")
    >>> await pipeline.run_async(file_path="meeting.wav")
"""

from vertector_data_ingestion.integrations.neo4j.loaders import (
    MultimodalLoader,
    VertectorAudioLoader,
    VertectorDataLoader,
)
from vertector_data_ingestion.integrations.neo4j.splitters import VertectorTextSplitter

__all__ = [
    "VertectorDataLoader",
    "VertectorAudioLoader",
    "VertectorTextSplitter",
    "MultimodalLoader",
]
