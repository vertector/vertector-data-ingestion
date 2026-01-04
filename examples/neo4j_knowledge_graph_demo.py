"""Production-ready Neo4j Knowledge Graph construction demo.

This script demonstrates building a knowledge graph from multimodal data
(documents and audio) using Vertector's Neo4j integration with SimpleKGPipeline.

Prerequisites:
    1. Neo4j running (use docker-compose.neo4j.yml)
    2. Ollama running with gemma2:4b model
    3. Install: uv pip install vertector-data-ingestion[neo4j] langchain-ollama

Usage:
    # Start Neo4j
    docker-compose -f docker-compose.neo4j.yml up -d

    # Start Ollama and pull model
    ollama serve
    ollama pull gemma2:4b

    # Run demo
    uv run examples/neo4j_knowledge_graph_demo.py

    # View graph
    Open http://localhost:7474 (neo4j/vertector_demo_2024)
"""

import asyncio
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from loguru import logger
from neo4j import GraphDatabase
from neo4j_graphrag.experimental.components.schema import (
    NodeType,
    RelationshipType,
    PropertyType,
)
from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import LLMInterface
from neo4j_graphrag.embeddings.base import Embedder
from sentence_transformers import SentenceTransformer

from vertector_data_ingestion.integrations.neo4j import (
    MultimodalLoader,
    VertectorTextSplitter,
)


class LangChainLLMWrapper(LLMInterface):
    """Wrapper to make LangChain LLMs compatible with Neo4j GraphRAG."""

    def __init__(self, llm: ChatOllama):
        self.llm = llm

    def invoke(self, prompt: str):
        """Synchronous invoke the LLM."""
        return self.llm.invoke(prompt)

    async def ainvoke(self, prompt: str):
        """Async invoke the LLM."""
        return await self.llm.ainvoke(prompt)


class SentenceTransformerEmbedder(Embedder):
    """Wrapper to make SentenceTransformer compatible with Neo4j GraphRAG."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-0.6B"):
        """Initialize SentenceTransformer embedder.

        Args:
            model_name: HuggingFace model name for embeddings
        """
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query (synchronous)."""
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents (synchronous)."""
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()


class Neo4jKnowledgeGraphBuilder:
    """Production-ready knowledge graph builder using Vertector + Neo4j + Ollama."""

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "vertector_demo_2024",
        ollama_model: str = "gemma3:4b",
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        chunk_size: int = 512,
        ollama_base_url: str = "http://localhost:11434",
    ):
        """Initialize knowledge graph builder.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            ollama_model: Ollama model for entity/relation extraction
            embedding_model: HuggingFace embedding model name
            chunk_size: Maximum tokens per chunk
            ollama_base_url: Ollama server URL
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.chunk_size = chunk_size

        # Load environment variables
        load_dotenv()

        # Initialize Ollama LLM (local, no API key needed)
        ollama_llm = ChatOllama(
            model=ollama_model,
            base_url=ollama_base_url,
            temperature=0.0,  # Deterministic for entity extraction
        )

        # Wrap for Neo4j compatibility
        self.llm = LangChainLLMWrapper(ollama_llm)

        # Initialize SentenceTransformer embeddings (local, no API calls)
        self.embedder = SentenceTransformerEmbedder(model_name=embedding_model)

        # Initialize Vertector components
        self.driver = None
        self.loader = MultimodalLoader()
        self.splitter = VertectorTextSplitter(loader=self.loader, chunk_size=chunk_size)
        self.pipeline = None

        logger.info("Knowledge graph builder initialized")
        logger.info(f"  LLM: {ollama_model} (Ollama)")
        logger.info(f"  Embedder: {embedding_model} (SentenceTransformer)")
        logger.info(f"  Chunk size: {chunk_size} tokens")

    def connect(self) -> None:
        """Connect to Neo4j database and verify connection."""
        logger.info(f"Connecting to Neo4j at {self.neo4j_uri}")

        self.driver = GraphDatabase.driver(
            self.neo4j_uri,
            auth=(self.neo4j_user, self.neo4j_password),
        )

        # Verify connection
        try:
            with self.driver.session() as session:
                result = session.run("RETURN 1 as test")
                result.single()
                logger.success("✓ Connected to Neo4j")
        except Exception as e:
            logger.error(f"Failed to connect to Neo4j: {e}")
            logger.info("Make sure Neo4j is running: docker-compose -f docker-compose.neo4j.yml up -d")
            raise

    def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def define_schema(
        self,
        node_types: list[NodeType],
        relationship_types: list[RelationshipType],
        patterns: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """Define knowledge graph schema using modern 2026 API.

        Args:
            node_types: List of NodeType objects defining entity types
            relationship_types: List of RelationshipType objects
            patterns: Optional list of (source, relation, target) tuples

        Example:
            >>> builder.define_schema(
            ...     node_types=[
            ...         NodeType(
            ...             label="Person",
            ...             properties=[
            ...                 PropertyType(name="name", type="STRING"),
            ...                 PropertyType(name="role", type="STRING"),
            ...             ]
            ...         ),
            ...     ],
            ...     relationship_types=[
            ...         RelationshipType(label="WORKS_AT"),
            ...     ],
            ...     patterns=[("Person", "WORKS_AT", "Organization")]
            ... )
        """
        logger.info("Defining knowledge graph schema")
        logger.info(f"  Node types: {[n.label for n in node_types]}")
        logger.info(f"  Relationship types: {[r.label for r in relationship_types]}")

        # Build schema dictionary for SimpleKGPipeline (2026 API)
        schema_config = {
            "node_types": node_types,
            "relationship_types": relationship_types,
            "additional_node_types": False,  # Strict schema enforcement
        }

        if patterns:
            schema_config["patterns"] = patterns
            logger.info(f"  Patterns: {len(patterns)} defined")

        # Create SimpleKGPipeline with custom Vertector components
        self.pipeline = SimpleKGPipeline(
            llm=self.llm,
            driver=self.driver,
            embedder=self.embedder,
            pdf_loader=self.loader,  # Custom MultimodalLoader
            text_splitter=self.splitter,  # Custom VertectorTextSplitter
            schema=schema_config,  # Modern 2026 API
            from_pdf=True,  # Use custom loader and splitter
            on_error="RAISE",
        )

        logger.success("✓ Schema defined, pipeline ready")

    async def process_file(self, file_path: Path | str) -> dict[str, Any]:
        """Process a single file and extract knowledge graph.

        Args:
            file_path: Path to document or audio file

        Returns:
            Dictionary containing extraction results:
            - file: Filename
            - chunks: Number of chunks created
            - entities: Number of entities extracted
            - relations: Number of relations extracted
            - entity_types: Entity type breakdown
            - relation_types: Relation type breakdown

        Raises:
            RuntimeError: If schema not defined (call define_schema first)
            FileNotFoundError: If file doesn't exist
        """
        if not self.pipeline:
            msg = "Schema not defined. Call define_schema() first."
            raise RuntimeError(msg)

        file_path = Path(file_path)
        if not file_path.exists():
            msg = f"File not found: {file_path}"
            raise FileNotFoundError(msg)

        logger.info(f"Processing {file_path.name}")

        # Run pipeline
        pipeline_result = await self.pipeline.run_async(file_path=str(file_path))

        # Extract statistics from PipelineResult (has run_id and result fields)
        # The actual data is in pipeline_result.result
        result = pipeline_result.result if hasattr(pipeline_result, 'result') else {}

        chunks = result.get("chunks", []) if isinstance(result, dict) else []
        entities = result.get("entities", []) if isinstance(result, dict) else []
        relations = result.get("relations", []) if isinstance(result, dict) else []

        stats = {
            "file": file_path.name,
            "chunks": len(chunks) if chunks else 0,
            "entities": len(entities) if entities else 0,
            "relations": len(relations) if relations else 0,
            "entity_types": {},
            "relation_types": {},
        }

        # Count entity types
        for entity in entities:
            label = entity.get("label", "Unknown") if isinstance(entity, dict) else "Unknown"
            stats["entity_types"][label] = stats["entity_types"].get(label, 0) + 1

        # Count relation types
        for relation in relations:
            rel_type = relation.get("type", "Unknown") if isinstance(relation, dict) else "Unknown"
            stats["relation_types"][rel_type] = stats["relation_types"].get(rel_type, 0) + 1

        logger.success(f"✓ {file_path.name}: {stats['entities']} entities, {stats['relations']} relations")

        return stats

    async def process_directory(self, directory: Path | str, pattern: str = "*") -> dict[str, Any]:
        """Process all matching files in a directory.

        Args:
            directory: Path to directory
            pattern: Glob pattern for files (default: all files)

        Returns:
            Dictionary containing aggregated statistics

        Example:
            >>> stats = await builder.process_directory("docs/", "*.pdf")
        """
        directory = Path(directory)
        if not directory.is_dir():
            msg = f"Not a directory: {directory}"
            raise ValueError(msg)

        # Find matching files
        files = list(directory.glob(pattern))
        if not files:
            logger.warning(f"No files matching '{pattern}' in {directory}")
            return {"files": 0, "total_entities": 0, "total_relations": 0}

        logger.info(f"Processing {len(files)} files from {directory}")

        # Process each file
        all_stats = []
        for file_path in files:
            try:
                stats = await self.process_file(file_path)
                all_stats.append(stats)
            except Exception as e:
                logger.error(f"Error processing {file_path.name}: {e}")

        # Aggregate statistics
        total_stats = {
            "files": len(all_stats),
            "total_entities": sum(s["entities"] for s in all_stats),
            "total_relations": sum(s["relations"] for s in all_stats),
            "total_chunks": sum(s["chunks"] for s in all_stats),
            "entity_types": {},
            "relation_types": {},
        }

        # Merge entity/relation type counts
        for stats in all_stats:
            for entity_type, count in stats["entity_types"].items():
                total_stats["entity_types"][entity_type] = (
                    total_stats["entity_types"].get(entity_type, 0) + count
                )
            for rel_type, count in stats["relation_types"].items():
                total_stats["relation_types"][rel_type] = (
                    total_stats["relation_types"].get(rel_type, 0) + count
                )

        return total_stats

    def query_graph(self, cypher_query: str) -> list[dict]:
        """Execute a Cypher query against the knowledge graph.

        Args:
            cypher_query: Cypher query string

        Returns:
            List of result records as dictionaries

        Example:
            >>> results = builder.query_graph(
            ...     "MATCH (p:Person)-[r:WORKS_AT]->(o:Organization) "
            ...     "RETURN p.name, o.name LIMIT 10"
            ... )
        """
        if not self.driver:
            msg = "Not connected to Neo4j. Call connect() first."
            raise RuntimeError(msg)

        with self.driver.session() as session:
            result = session.run(cypher_query)
            records = [record.data() for record in result]
            return records


# Example schema for research papers (2026 API)
RESEARCH_PAPER_NODE_TYPES = [
    NodeType(
        label="Person",
        properties=[
            PropertyType(name="name", type="STRING"),
            PropertyType(name="affiliation", type="STRING"),
            PropertyType(name="role", type="STRING"),
        ],
    ),
    NodeType(
        label="Organization",
        properties=[
            PropertyType(name="name", type="STRING"),
            PropertyType(name="type", type="STRING"),
        ],
    ),
    NodeType(
        label="Technology",
        properties=[
            PropertyType(name="name", type="STRING"),
            PropertyType(name="category", type="STRING"),
        ],
    ),
    NodeType(
        label="ResearchTopic",
        properties=[
            PropertyType(name="name", type="STRING"),
            PropertyType(name="field", type="STRING"),
        ],
    ),
    NodeType(
        label="Dataset",
        properties=[
            PropertyType(name="name", type="STRING"),
            PropertyType(name="size", type="STRING"),
        ],
    ),
]

RESEARCH_PAPER_RELATIONSHIP_TYPES = [
    RelationshipType(label="WORKS_AT"),
    RelationshipType(label="RESEARCHES"),
    RelationshipType(label="USES"),
    RelationshipType(label="IMPLEMENTS"),
    RelationshipType(label="EVALUATES_ON"),
    RelationshipType(label="COLLABORATES_WITH"),
]

RESEARCH_PAPER_PATTERNS = [
    ("Person", "WORKS_AT", "Organization"),
    ("Person", "RESEARCHES", "ResearchTopic"),
    ("Person", "USES", "Technology"),
    ("Person", "EVALUATES_ON", "Dataset"),
    ("Person", "COLLABORATES_WITH", "Person"),
    ("Technology", "IMPLEMENTS", "Technology"),
]


async def demo():
    """Run complete knowledge graph construction demo."""
    logger.info("=" * 60)
    logger.info("Vertector + Neo4j + Ollama Knowledge Graph Demo")
    logger.info("=" * 60)

    # Initialize builder
    builder = Neo4jKnowledgeGraphBuilder(
        ollama_model="gemma3:4b",  # Local Ollama LLM
        embedding_model="Qwen/Qwen3-Embedding-0.6B",  # Local SentenceTransformer
        chunk_size=512,
    )

    try:
        # Connect to Neo4j (synchronous)
        builder.connect()

        # Define schema (2026 API)
        builder.define_schema(
            node_types=RESEARCH_PAPER_NODE_TYPES,
            relationship_types=RESEARCH_PAPER_RELATIONSHIP_TYPES,
            patterns=RESEARCH_PAPER_PATTERNS,
        )

        # Process sample files
        test_docs_dir = Path(__file__).parent.parent / "test_documents"

        logger.info("\n" + "=" * 60)
        logger.info("Processing Documents")
        logger.info("=" * 60)

        # Process PDF
        pdf_file = test_docs_dir / "2112.13734v2.pdf"
        if pdf_file.exists():
            pdf_stats = await builder.process_file(pdf_file)
            logger.info(f"\nPDF Statistics:")
            logger.info(f"  Chunks: {pdf_stats['chunks']}")
            logger.info(f"  Entities: {pdf_stats['entities']}")
            logger.info(f"  Relations: {pdf_stats['relations']}")
            logger.info(f"  Entity types: {pdf_stats['entity_types']}")
            logger.info(f"  Relation types: {pdf_stats['relation_types']}")
        else:
            logger.warning(f"PDF file not found: {pdf_file}")

        # Process Audio
        logger.info("\n" + "=" * 60)
        logger.info("Processing Audio")
        logger.info("=" * 60)

        audio_file = test_docs_dir / "harvard.wav"
        if audio_file.exists():
            audio_stats = await builder.process_file(audio_file)
            logger.info(f"\nAudio Statistics:")
            logger.info(f"  Chunks: {audio_stats['chunks']}")
            logger.info(f"  Entities: {audio_stats['entities']}")
            logger.info(f"  Relations: {audio_stats['relations']}")
            logger.info(f"  Entity types: {audio_stats['entity_types']}")
            logger.info(f"  Relation types: {audio_stats['relation_types']}")
        else:
            logger.warning(f"Audio file not found: {audio_file}")

        # Query the graph
        logger.info("\n" + "=" * 60)
        logger.info("Querying Knowledge Graph")
        logger.info("=" * 60)

        # Example query: Get all entities
        query = "MATCH (n) RETURN labels(n)[0] as type, count(n) as count ORDER BY count DESC"
        results = builder.query_graph(query)
        logger.info("\nEntity counts:")
        for record in results:
            logger.info(f"  {record['type']}: {record['count']}")

        # Example query: Get sample relationships
        query = """
        MATCH (a)-[r]->(b)
        RETURN labels(a)[0] as source_type, type(r) as relation, labels(b)[0] as target_type, count(*) as count
        ORDER BY count DESC
        LIMIT 10
        """
        results = builder.query_graph(query)
        logger.info("\nTop relationships:")
        for record in results:
            logger.info(f"  {record['source_type']} -{record['relation']}-> {record['target_type']}: {record['count']}")

        logger.info("\n" + "=" * 60)
        logger.success("✓ Demo Complete!")
        logger.info("=" * 60)
        logger.info("\nView the knowledge graph:")
        logger.info("  1. Open http://localhost:7474 in your browser")
        logger.info("  2. Login: neo4j / vertector_demo_2024")
        logger.info("  3. Try queries:")
        logger.info("     MATCH (n) RETURN n LIMIT 100")
        logger.info("     MATCH p=()-[]->() RETURN p LIMIT 100")

    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise
    finally:
        builder.close()


if __name__ == "__main__":
    # Configure logging
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    # Run demo
    asyncio.run(demo())
