"""Knowledge Graph Builder Demo - High-level API for multimodal knowledge graphs.

This example demonstrates the KnowledgeGraphBuilder API which provides:
- Unified process() method for files, directories, or combinations
- Automatic handling of documents (PDF, DOCX, etc.) and audio (WAV, MP3, etc.)
- Built-in Vertector config integration (LocalMpsConfig, CloudGpuConfig, etc.)
- Neo4j SimpleKGPipeline integration

Requirements:
    - Neo4j running: docker-compose -f docker-compose.neo4j.yml up -d
    - Ollama with model: ollama pull gemma3:4b
"""

import asyncio
from pathlib import Path

from neo4j_graphrag.experimental.components.schema import (
    NodeType,
    PropertyType,
    RelationshipType,
)

from vertector_data_ingestion import LocalMpsConfig, setup_logging
from vertector_data_ingestion.integrations.neo4j import KnowledgeGraphBuilder

# Setup logging
setup_logging(log_level="INFO")


async def main():
    """Demonstrate KnowledgeGraphBuilder with the unified process() API."""

    print("=" * 80)
    print("KNOWLEDGE GRAPH BUILDER DEMO")
    print("=" * 80)

    # 1. Initialize builder
    print("\n[1/5] Initializing KnowledgeGraphBuilder...")
    builder = KnowledgeGraphBuilder(
        neo4j_uri="bolt://localhost:7687",
        neo4j_password="vertector_demo_2024",
        ollama_model="gemma3:4b",
        embedding_model="Qwen/Qwen3-Embedding-0.6B",
        converter_config=LocalMpsConfig(),  # Reuses existing Vertector config
        chunk_size=512,
    )
    print("✓ Builder initialized")

    # 2. Connect to Neo4j
    print("\n[2/5] Connecting to Neo4j...")
    builder.connect()
    print("✓ Connected")

    # 3. Define schema
    print("\n[3/5] Defining knowledge graph schema...")

    node_types = [
        NodeType(
            label="Author",
            properties=[
                PropertyType(name="name", type="STRING"),
                PropertyType(name="affiliation", type="STRING"),
            ],
        ),
        NodeType(
            label="Paper",
            properties=[
                PropertyType(name="title", type="STRING"),
                PropertyType(name="year", type="INTEGER"),
            ],
        ),
        NodeType(
            label="Concept",
            properties=[PropertyType(name="name", type="STRING")],
        ),
    ]

    relationship_types = [
        RelationshipType(label="AUTHORED"),
        RelationshipType(label="CITES"),
        RelationshipType(label="DISCUSSES"),
    ]

    patterns = [
        ("Author", "AUTHORED", "Paper"),
        ("Paper", "CITES", "Paper"),
        ("Paper", "DISCUSSES", "Concept"),
    ]

    builder.define_schema(
        node_types=node_types,
        relationship_types=relationship_types,
        patterns=patterns,
    )
    print("✓ Schema defined")

    # 4. Process files using the unified process() method
    print("\n[4/5] Processing files with unified process() API...")

    # Example 1: Single file (string, not Path)
    print("\n  Example 1: Single file")
    if Path("test_documents/sample1.pdf").exists():
        stats = await builder.process("test_documents/sample1.pdf")
        print(f"    Files: {stats['files_processed']}")
        print(f"    Entities: {stats['total_entities']}")
        print(f"    Relations: {stats['total_relations']}")

    # 5. Query the knowledge graph
    print("\n[5/5] Querying knowledge graph...")

    queries = [
        ("Total nodes", "MATCH (n) RETURN count(n) as count"),
        ("Total relationships", "MATCH ()-[r]->() RETURN count(r) as count"),
        (
            "Node types",
            "MATCH (n) RETURN DISTINCT labels(n)[0] as type, count(n) as count ORDER BY count DESC LIMIT 5",
        ),
    ]

    for name, cypher in queries:
        results = builder.query_graph(cypher)
        print(f"\n  {name}:")
        for result in results:
            print(f"    {result}")

    # Cleanup
    builder.close()
    print("\n" + "=" * 80)
    print("✓ DEMO COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
