"""Configurable Knowledge Graph Builder for Neo4j.

This module provides a production-ready, configurable interface for building
knowledge graphs from multimodal data using Neo4j SimpleKGPipeline.

Example:
    >>> from vertector_data_ingestion.integrations.neo4j import KnowledgeGraphBuilder
    >>> from vertector_data_ingestion import LocalMpsConfig
    >>> from neo4j_graphrag.experimental.components.schema import NodeType, RelationshipType, PropertyType
    >>>
    >>> # Create builder with custom config
    >>> config = LocalMpsConfig()
    >>> builder = KnowledgeGraphBuilder(
    ...     neo4j_uri="bolt://localhost:7687",
    ...     neo4j_user="neo4j",
    ...     neo4j_password="password",
    ...     ollama_model="gemma3:4b",
    ...     embedding_model="Qwen/Qwen3-Embedding-0.6B",
    ...     converter_config=config,
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
    >>> # Process files (accepts string or list of strings)
    >>> stats = await builder.process("document.pdf")
    >>> # Or multiple files/directories
    >>> stats = await builder.process(["doc1.pdf", "documents/", "audio.wav"])
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Optional

from loguru import logger
from neo4j import GraphDatabase
from neo4j_graphrag.embeddings.base import Embedder
from neo4j_graphrag.experimental.components.schema import (
    NodeType,
    RelationshipType,
)
from neo4j_graphrag.experimental.components.types import LexicalGraphConfig

from neo4j_graphrag.experimental.pipeline.kg_builder import SimpleKGPipeline
from neo4j_graphrag.llm import LLMInterface
from sentence_transformers import SentenceTransformer

from vertector_data_ingestion.integrations.neo4j.loaders import MultimodalLoader
from vertector_data_ingestion.integrations.neo4j.merge_kg_writer import MergeKGWriter
from vertector_data_ingestion.integrations.neo4j.splitters import VertectorTextSplitter
from vertector_data_ingestion.models.config import ConverterConfig, LocalMpsConfig


class LangChainLLMWrapper(LLMInterface):
    """Wrapper to make LangChain LLMs compatible with Neo4j GraphRAG."""

    def __init__(self, llm: Any):
        """Initialize wrapper with a LangChain LLM.

        Args:
            llm: LangChain LLM instance (e.g., ChatOllama, ChatOpenAI)
        """
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


class KnowledgeGraphBuilder:
    """Production-ready knowledge graph builder with configurable multimodal support.

    This class provides a high-level interface for building knowledge graphs from
    documents and audio files using Vertector's processing pipelines and Neo4j.

    Attributes:
        converter_config: Vertector configuration for document/audio processing
        neo4j_uri: Neo4j connection URI
        neo4j_user: Neo4j username
        neo4j_password: Neo4j password
        ollama_model: Ollama model name for entity extraction
        embedding_model: Embedding model name
        chunk_size: Override chunk size (uses converter_config.chunking.size if None)
    """

    def __init__(
        self,
        neo4j_uri: str = "bolt://localhost:7687",
        neo4j_user: str = "neo4j",
        neo4j_password: str = "vertector_demo_2024",
        ollama_model: str = "gemma3:4b",
        ollama_base_url: str = "http://localhost:11434",
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        converter_config: Optional[ConverterConfig] = None,
        chunk_size: Optional[int] = None,
        on_error: str = "RAISE",
    ):
        """Initialize knowledge graph builder.

        Args:
            neo4j_uri: Neo4j connection URI
            neo4j_user: Neo4j username
            neo4j_password: Neo4j password
            ollama_model: Ollama model for entity/relation extraction
            ollama_base_url: Ollama server URL
            embedding_model: Embedding model (SentenceTransformer compatible)
            converter_config: Vertector config (defaults to LocalMpsConfig)
            chunk_size: Override chunk size (uses config.chunking.size if None)
            on_error: Error handling: "RAISE" or "IGNORE"
        """
        self.neo4j_uri = neo4j_uri
        self.neo4j_user = neo4j_user
        self.neo4j_password = neo4j_password
        self.ollama_model = ollama_model
        self.ollama_base_url = ollama_base_url
        self.embedding_model = embedding_model
        self.on_error = on_error

        # Use provided config or default to LocalMpsConfig
        self.converter_config = converter_config or LocalMpsConfig()

        # Chunk size: explicit override > config value > default
        self.chunk_size = (
            chunk_size
            if chunk_size is not None
            else self.converter_config.chunking.size
        )

        self.driver = None
        self.pipeline = None

        # Initialize loader with converter config
        self.loader = MultimodalLoader(
            vertector_config=self.converter_config,
            audio_config=self.converter_config.audio,
        )

        # Initialize splitter with chunk size
        self.splitter = VertectorTextSplitter(
            loader=self.loader,
            chunk_size=self.chunk_size,
        )

        # LLM and embedder will be initialized lazily
        self.llm = None
        self.embedder = None

        logger.info("KnowledgeGraphBuilder initialized")
        logger.info(f"  LLM: {self.ollama_model} (Ollama)")
        logger.info(f"  Embedder: {self.embedding_model}")
        logger.info(f"  Chunk size: {self.chunk_size} tokens")
        logger.info(f"  Config: {type(self.converter_config).__name__}")

    def _initialize_llm(self) -> LLMInterface:
        """Initialize LLM (lazy initialization).

        Returns:
            Wrapped LLM compatible with Neo4j GraphRAG
        """
        if self.llm is None:
            from langchain_ollama import ChatOllama

            ollama_llm = ChatOllama(
                model=self.ollama_model,
                base_url=self.ollama_base_url,
                temperature=0.0,  # Deterministic for entity extraction
            )
            self.llm = LangChainLLMWrapper(ollama_llm)
            logger.info(f"Initialized LLM: {self.ollama_model}")

        return self.llm

    def _initialize_embedder(self) -> Embedder:
        """Initialize embedder (lazy initialization).

        Returns:
            Embedder compatible with Neo4j GraphRAG
        """
        if self.embedder is None:
            self.embedder = SentenceTransformerEmbedder(model_name=self.embedding_model)
            logger.info(f"Initialized embedder: {self.embedding_model}")

        return self.embedder

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
            logger.info(
                "Make sure Neo4j is running: docker-compose -f docker-compose.neo4j.yml up -d"
            )
            raise

    def close(self) -> None:
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()
            logger.info("Neo4j connection closed")

    def create_constraints(
        self, 
        entity_labels: list[str],
        lexical_graph_config: LexicalGraphConfig,  # No longer Optional - always provided
    ) -> None:
        """Create unique constraints for complete idempotency across all node types.
        
        Creates constraints for:
        1. Entity nodes (e.g., Person, Organization) - unique by 'name' property
        2. Document nodes - unique by 'path' property
        3. Chunk nodes - unique by 'id' property
        
        This ensures that running the pipeline multiple times on the same data
        doesn't create duplicate nodes of ANY type.
        
        Args:
            entity_labels: List of entity label strings (e.g., ["Person", "Organization"])
            lexical_graph_config: LexicalGraphConfig with document/chunk node labels
        
        Example:
            >>> builder.connect()
            >>> builder.create_constraints(["Person", "Organization"])
            >>> builder.define_schema(...)
        """
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")
        
        # Get lexical graph labels from config (defaults: "Document" and "Chunk")
        doc_label = lexical_graph_config.document_node_label
        chunk_label = lexical_graph_config.chunk_node_label
        
        logger.info("Creating unique constraints for complete idempotency")
        logger.info(f"  Entity types: {len(entity_labels)}")
        logger.info(f"  Lexical graph: {doc_label}, {chunk_label}")
        
        with self.driver.session() as session:
            # 1. Create constraints for entity nodes (unique by 'name' property)
            for label in entity_labels:
                try:
                    constraint_name = f"{label.lower()}_name_unique"
                    query = f"""
                    CREATE CONSTRAINT {constraint_name} IF NOT EXISTS
                    FOR (n:{label})
                    REQUIRE n.name IS UNIQUE
                    """
                    session.run(query)
                    logger.debug(f"✓ Entity constraint: {label}.name IS UNIQUE")
                except Exception as e:
                    logger.debug(f"Constraint for {label} may already exist: {e}")
            
            # 2. Create constraint for Document nodes (unique by 'path' property)
            try:
                doc_constraint_name = f"{doc_label.lower()}_path_unique"
                doc_query = f"""
                CREATE CONSTRAINT {doc_constraint_name} IF NOT EXISTS
                FOR (d:{doc_label})
                REQUIRE d.path IS UNIQUE
                """
                session.run(doc_query)
                logger.debug(f"✓ Document constraint: {doc_label}.path IS UNIQUE")
            except Exception as e:
                logger.debug(f"Constraint for {doc_label} may already exist: {e}")
            
            # 3. Create constraint for Chunk nodes (unique by 'id' property)
            try:
                chunk_constraint_name = f"{chunk_label.lower()}_id_unique"
                chunk_query = f"""
                CREATE CONSTRAINT {chunk_constraint_name} IF NOT EXISTS
                FOR (c:{chunk_label})
                REQUIRE c.id IS UNIQUE
                """
                session.run(chunk_query)
                logger.debug(f"✓ Chunk constraint: {chunk_label}.id IS UNIQUE")
            except Exception as e:
                logger.debug(f"Constraint for {chunk_label} may already exist: {e}")
        
        logger.success(
            f"✓ Complete idempotency: {len(entity_labels)} entity types + "
            f"{doc_label} + {chunk_label} constraints created"
        )
        
    def define_schema(
        self,
        node_types: list[NodeType],
        relationship_types: list[RelationshipType],
        patterns: list[tuple[str, str, str]] | None = None,
        lexical_graph_config: LexicalGraphConfig | None = None,  # Optional with smart default
        prompt_template: str | None = None,  # Custom extraction prompt template
    ) -> None:
        """Define knowledge graph schema using Neo4j GraphRAG API.

        Args:
            node_types: List of NodeType objects defining entity types
            relationship_types: List of RelationshipType objects
            patterns: Optional list of (source, relation, target) tuples
            lexical_graph_config: Optional LexicalGraphConfig. If None, uses default
                                (Document and Chunk labels). Only pass this if you need
                                custom lexical graph node labels.
            prompt_template: Optional custom prompt template for entity/relationship extraction.
                           Template can use {text} and {schema} placeholders.
                           If None, uses default ERExtractionTemplate.

        Example:
            >>> from neo4j_graphrag.experimental.components.schema import (
            ...     NodeType, RelationshipType, PropertyType
            ... )
            >>> # Example 1: Basic usage with default prompt
            >>> builder.define_schema(
            ...     node_types=[
            ...         NodeType(
            ...             label="Person",
            ...             properties=[
            ...                 PropertyType(name="name", type="STRING"),
            ...                 PropertyType(name="age", type="INTEGER"),
            ...             ]
            ...         ),
            ...     ],
            ...     relationship_types=[RelationshipType(label="KNOWS")],
            ...     patterns=[("Person", "KNOWS", "Person")]
            ... )
            >>>
            >>> # Example 2: With custom prompt template
            >>> custom_prompt = '''Extract entities from the text.
            ...
            ... TEXT:
            ... {text}
            ...
            ... SCHEMA:
            ... {schema}
            ... '''
            >>> builder.define_schema(
            ...     node_types=[NodeType(label="Person")],
            ...     relationship_types=[RelationshipType(label="KNOWS")],
            ...     prompt_template=custom_prompt
            ... )
            >>> # Constraints created automatically for Person + Document + Chunk
        """
        logger.info("Defining knowledge graph schema")
        logger.info(f"  Node types: {[n.label for n in node_types]}")
        logger.info(f"  Relationship types: {[r.label for r in relationship_types]}")
        
        # Use default LexicalGraphConfig if not provided (matches SimpleKGPipeline default)
        if lexical_graph_config is None:
            lexical_graph_config = LexicalGraphConfig()
        
        # CRITICAL: Create unique constraints BEFORE initializing pipeline
        # This ensures idempotency for entities AND lexical graph (Document/Chunk)
        entity_labels = [node.label for node in node_types]
        self.create_constraints(entity_labels, lexical_graph_config)
        
        # Initialize LLM and embedder
        llm = self._initialize_llm()
        embedder = self._initialize_embedder()
        
        # Build schema dictionary
        schema_config = {
            "node_types": node_types,
            "relationship_types": relationship_types,
            "additional_node_types": False,  # Strict schema enforcement
        }
        
        if patterns:
            schema_config["patterns"] = patterns
            logger.info(f"  Patterns: {len(patterns)} defined")
        
        # Create custom MERGE-based KG writer for true idempotency
        # Default Neo4jWriter has a bug: uses apoc.create.addLabels which fails with constraints
        # MergeKGWriter uses pure Cypher MERGE operations that work with constraints
        kg_writer = MergeKGWriter(driver=self.driver)

        # Build SimpleKGPipeline parameters
        pipeline_params = {
            "llm": llm,
            "driver": self.driver,
            "embedder": embedder,
            "pdf_loader": self.loader,  # Custom MultimodalLoader
            "text_splitter": self.splitter,  # Custom VertectorTextSplitter
            "kg_writer": kg_writer,  # Custom MERGE-based writer (fixes APOC constraint bug)
            "schema": schema_config,
            "from_pdf": True,  # Use custom loader and splitter
            "on_error": self.on_error,
            "perform_entity_resolution": True,  # Entity resolution merges duplicates
            "lexical_graph_config": lexical_graph_config,  # Use same config for consistency
        }

        # Add custom prompt template if provided
        if prompt_template:
            pipeline_params["prompt_template"] = prompt_template
            logger.info("  Using custom prompt template for entity/relationship extraction")

        # Create SimpleKGPipeline with MergeKGWriter + constraints = complete idempotency
        self.pipeline = SimpleKGPipeline(**pipeline_params)
        
        logger.success("✓ Schema + constraints defined, pipeline ready")

    async def process(
        self,
        paths: str | list[str],
        pattern: str = "**/*",
        recursive: bool = True
    ) -> dict[str, Any]:
        """Intelligently process files, directories, or any combination.

        This method handles:
        - Single file: "document.pdf"
        - List of files: ["doc1.pdf", "doc2.pdf"]
        - Single directory: "documents/"
        - List of directories: ["docs1/", "docs2/"]
        - Mixed combination: ["doc.pdf", "documents/", "audio.wav", "more_docs/"]

        Args:
            paths: Single path string, or list of path strings
            pattern: Glob pattern for directory processing (default: "**/*")
            recursive: Whether to process directories recursively (default: True)

        Returns:
            Dictionary containing aggregated extraction results

        Raises:
            RuntimeError: If schema not defined
            ValueError: If paths contain Path objects instead of strings

        Examples:
            >>> # Single file
            >>> stats = await builder.process("document.pdf")
            >>>
            >>> # Multiple files
            >>> stats = await builder.process(["doc1.pdf", "doc2.pdf", "audio.wav"])
            >>>
            >>> # Directory with pattern
            >>> stats = await builder.process("documents/", pattern="**/*.pdf")
            >>>
            >>> # Mixed: files + directories
            >>> stats = await builder.process([
            ...     "important.pdf",
            ...     "documents/",
            ...     "audio_files/",
            ...     "summary.docx"
            ... ])
        """
        if not self.pipeline:
            msg = "Schema not defined. Call define_schema() first."
            raise RuntimeError(msg)

        # Normalize to list
        if isinstance(paths, str):
            paths = [paths]
        elif not isinstance(paths, list):
            msg = f"paths must be string or list of strings, got {type(paths)}"
            raise ValueError(msg)

        # Validate all paths are strings
        for p in paths:
            if not isinstance(p, str):
                msg = f"All paths must be strings, found {type(p)}. Use: builder.process('file.pdf')"
                raise ValueError(msg)

        logger.info(f"Processing {len(paths)} path(s)")

        # Process all paths and collect results
        all_file_stats = []
        failed_items = []
        total_files_processed = 0
        total_dirs_processed = 0

        for path_str in paths:
            path_obj = Path(path_str)

            if not path_obj.exists():
                logger.warning(f"Path not found, skipping: {path_str}")
                failed_items.append({"path": path_str, "error": "Path not found"})
                continue

            try:
                if path_obj.is_file():
                    # Process single file
                    logger.info(f"Processing file: {path_obj.name}")
                    stats = await self._process_single_file(path_obj)
                    all_file_stats.append(stats)
                    total_files_processed += 1

                elif path_obj.is_dir():
                    # Process directory
                    logger.info(f"Processing directory: {path_obj}")
                    dir_pattern = pattern if recursive else pattern.lstrip("**/")
                    dir_stats = await self._process_directory(path_obj, dir_pattern)

                    # Extract individual file stats from directory result
                    if "file_stats" in dir_stats:
                        all_file_stats.extend(dir_stats["file_stats"])

                    total_files_processed += dir_stats.get("files_processed", 0)
                    total_dirs_processed += 1

                    # Track directory-level failures
                    if dir_stats.get("failed_files"):
                        failed_items.extend(dir_stats["failed_files"])

            except Exception as e:
                logger.error(f"Error processing {path_str}: {e}")
                failed_items.append({"path": path_str, "error": str(e)})

        # Aggregate all statistics
        aggregated_stats = {
            "type": "batch" if len(paths) > 1 else ("file" if Path(paths[0]).is_file() else "directory"),
            "input_paths": len(paths),
            "directories_processed": total_dirs_processed,
            "files_processed": total_files_processed,
            "files_failed": len(failed_items),
            "total_chunks": sum(s.get("chunks", 0) for s in all_file_stats),
            "total_entities": sum(s.get("entities", 0) for s in all_file_stats),
            "total_relations": sum(s.get("relations", 0) for s in all_file_stats),
            "entity_types": {},
            "relation_types": {},
            "failed_items": failed_items if failed_items else None,
        }

        # Aggregate entity and relation types across all files
        for stats in all_file_stats:
            for entity_type, count in stats.get("entity_types", {}).items():
                aggregated_stats["entity_types"][entity_type] = (
                    aggregated_stats["entity_types"].get(entity_type, 0) + count
                )
            for rel_type, count in stats.get("relation_types", {}).items():
                aggregated_stats["relation_types"][rel_type] = (
                    aggregated_stats["relation_types"].get(rel_type, 0) + count
                )

        logger.success(
            f"✓ Batch complete: {aggregated_stats['files_processed']} files processed, "
            f"{aggregated_stats['total_entities']} entities, "
            f"{aggregated_stats['total_relations']} relations"
        )

        return aggregated_stats

    def _get_entity_stats(self) -> dict[str, Any]:
        """Query Neo4j for current entity and relationship statistics.

        Returns dict with entity_types and relation_types counts.
        """
        # Count entities by type (excluding internal __ labels)
        entity_query = """
        MATCH (n)
        WHERE n:`__Entity__`
        WITH DISTINCT labels(n) AS entity_labels
        UNWIND entity_labels AS entity_label
        WITH entity_label
        WHERE NOT entity_label STARTS WITH "__"
        WITH entity_label
        MATCH (n)
        WHERE entity_label IN labels(n) AND n:`__Entity__`
        RETURN entity_label as label, count(n) as count
        """
        entity_results = self.query_graph(entity_query)

        entity_stats = {}
        for row in entity_results:
            label = row.get("label")
            count = row.get("count", 0)
            if label:
                entity_stats[label] = count

        # Count relationships between __Entity__ nodes
        relation_query = """
        MATCH (start:`__Entity__`)-[r]->(end:`__Entity__`)
        RETURN type(r) as type, count(r) as count
        """
        relation_results = self.query_graph(relation_query)

        relation_stats = {}
        for row in relation_results:
            rel_type = row.get("type")
            count = row.get("count", 0)
            if rel_type:
                relation_stats[rel_type] = count

        return {
            "entity_types": entity_stats,
            "relation_types": relation_stats,
        }

    async def _process_single_file(self, file_path: Path) -> dict[str, Any]:
        """Internal: Process a single file.

        Args:
            file_path: Path object to file

        Returns:
            File-level statistics
        """
        logger.debug(f"Running pipeline on: {file_path.name}")

        # Snapshot entity/relation counts BEFORE processing
        before_stats = self._get_entity_stats()

        # Run Neo4j pipeline - writes directly to Neo4j
        pipeline_result = await self.pipeline.run_async(file_path=str(file_path))

        # Snapshot entity/relation counts AFTER processing
        after_stats = self._get_entity_stats()

        # Calculate the delta (entities/relations added by THIS file)
        entity_stats = {}
        total_entities = 0

        # Get all entity types (union of before and after)
        all_entity_types = set(before_stats["entity_types"].keys()) | set(after_stats["entity_types"].keys())
        for entity_type in all_entity_types:
            before_count = before_stats["entity_types"].get(entity_type, 0)
            after_count = after_stats["entity_types"].get(entity_type, 0)
            delta = after_count - before_count
            if delta > 0:
                entity_stats[entity_type] = delta
                total_entities += delta

        # Calculate relation delta
        relation_stats = {}
        total_relations = 0

        all_relation_types = set(before_stats["relation_types"].keys()) | set(after_stats["relation_types"].keys())
        for relation_type in all_relation_types:
            before_count = before_stats["relation_types"].get(relation_type, 0)
            after_count = after_stats["relation_types"].get(relation_type, 0)
            delta = after_count - before_count
            if delta > 0:
                relation_stats[relation_type] = delta
                total_relations += delta

        stats = {
            "type": "file",
            "path": str(file_path),
            "file": file_path.name,
            "chunks": 0,  # Not tracked separately
            "entities": total_entities,
            "relations": total_relations,
            "entity_types": entity_stats,
            "relation_types": relation_stats,
        }

        logger.success(
            f"✓ {file_path.name}: {stats['entities']} entities, {stats['relations']} relations"
        )

        return stats

    async def _process_directory(
        self, directory: Path, pattern: str = "**/*"
    ) -> dict[str, Any]:
        """Internal: Process all matching files in a directory.

        Args:
            directory: Path object to directory
            pattern: Glob pattern

        Returns:
            Directory-level aggregated statistics
        """
        files = list(directory.glob(pattern))
        files = [f for f in files if f.is_file()]  # Filter directories

        if not files:
            logger.warning(f"No files matching '{pattern}' in {directory}")
            return {
                "type": "directory",
                "path": str(directory),
                "files_processed": 0,
                "files_failed": 0,
                "file_stats": [],
                "failed_files": [],
            }

        logger.info(f"Found {len(files)} file(s) in {directory}")

        file_stats = []
        failed_files = []

        for file_path in files:
            try:
                stats = await self._process_single_file(file_path)
                file_stats.append(stats)
            except Exception as e:
                logger.error(f"Failed: {file_path.name} - {e}")
                failed_files.append({"path": str(file_path), "file": file_path.name, "error": str(e)})

        return {
            "type": "directory",
            "path": str(directory),
            "files_processed": len(file_stats),
            "files_failed": len(failed_files),
            "file_stats": file_stats,  # Return individual stats for aggregation
            "failed_files": failed_files,
        }

    def query_graph(self, cypher_query: str) -> list[dict]:
        """Execute a Cypher query against the knowledge graph.

        Args:
            cypher_query: Cypher query string

        Returns:
            List of result records as dictionaries
        """
        if not self.driver:
            raise RuntimeError("Not connected to Neo4j. Call connect() first.")

        with self.driver.session() as session:
            result = session.run(cypher_query)
            return [record.data() for record in result]

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


__all__ = [
    "KnowledgeGraphBuilder",
    "LangChainLLMWrapper",
    "SentenceTransformerEmbedder",
]
