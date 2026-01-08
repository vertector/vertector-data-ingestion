"""Neo4j SimpleKGPipeline integration components.

This module provides production-ready adapters for integrating Vertector Data Ingestion
with Neo4j's SimpleKGPipeline for multimodal knowledge graph construction.

Installation:
    uv pip install vertector-data-ingestion[neo4j]

Example - Using low-level components:
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

Example - Using high-level KnowledgeGraphBuilder:
    >>> from vertector_data_ingestion.integrations.neo4j import KnowledgeGraphBuilder
    >>> from vertector_data_ingestion import LocalMpsConfig
    >>> from neo4j_graphrag.experimental.components.schema import NodeType, RelationshipType
    >>>
    >>> # Create builder
    >>> builder = KnowledgeGraphBuilder(
    ...     neo4j_uri="bolt://localhost:7687",
    ...     neo4j_password="password",
    ...     ollama_model="gemma3:4b",
    ...     converter_config=LocalMpsConfig(),
    ... )
    >>>
    >>> # Connect and define schema
    >>> builder.connect()
    >>> builder.define_schema(
    ...     node_types=[NodeType(label="Person", properties=[...])],
    ...     relationship_types=[RelationshipType(label="KNOWS")],
    ...     patterns=[("Person", "KNOWS", "Person")]
    ... )
    >>>
    >>> # Process files
    >>> stats = await builder.process_file("document.pdf")
"""

from vertector_data_ingestion.integrations.neo4j.kg_builder import (
    KnowledgeGraphBuilder,
    LangChainLLMWrapper,
    SentenceTransformerEmbedder,
)
from vertector_data_ingestion.integrations.neo4j.loaders import (
    MultimodalLoader,
    VertectorAudioLoader,
    VertectorDataLoader,
)
from vertector_data_ingestion.integrations.neo4j.merge_kg_writer import MergeKGWriter
from vertector_data_ingestion.integrations.neo4j.content_hash import ContentHasher
from vertector_data_ingestion.integrations.neo4j.splitters import VertectorTextSplitter

__all__ = [
    # Low-level components
    "VertectorDataLoader",
    "VertectorAudioLoader",
    "VertectorTextSplitter",
    "MultimodalLoader",
    "MergeKGWriter",
    "ContentHasher",
    # High-level builder
    "KnowledgeGraphBuilder",
    "LangChainLLMWrapper",
    "SentenceTransformerEmbedder",
]
