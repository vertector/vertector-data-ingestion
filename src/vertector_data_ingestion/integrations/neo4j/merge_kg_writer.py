"""Custom MERGE-based KGWriter for idempotent knowledge graph construction.

This module provides a KGWriter implementation that uses MERGE instead of CREATE,
ensuring that running the pipeline multiple times on the same data doesn't create
duplicate entities. This is critical for production use where pipelines may be
re-run for updates or error recovery.

Key differences from default Neo4jWriter:
- Uses MERGE instead of CREATE for nodes
- Uses MERGE for relationships with apoc.merge.relationship
- Creates uniqueness constraints on entity names
- Ensures true idempotency across multiple pipeline runs

References:
    - https://neo4j.com/graphacademy/training-updating-40/04-updating40-merging-data/
    - https://neo4j.com/docs/neo4j-graphrag-python/current/user_guide_kg_builder.html
"""

from typing import Any

from loguru import logger
from neo4j import Driver
from pydantic import validate_call
from neo4j_graphrag.experimental.components.kg_writer import KGWriter
from neo4j_graphrag.experimental.components.types import (
    LexicalGraphConfig,
    Neo4jGraph,
)
from neo4j_graphrag.experimental.components.kg_writer import KGWriterModel

from vertector_data_ingestion.integrations.neo4j.content_hash import ContentHasher


class MergeKGWriter(KGWriter):
    """KGWriter that uses MERGE for idempotent entity creation.

    This writer ensures that entities are not duplicated when the pipeline
    is run multiple times. It uses MERGE operations with uniqueness constraints
    on the 'name' property (the default property used by entity resolution).

    Args:
        driver: Neo4j driver instance
        neo4j_database: Optional database name

    Example:
        >>> from neo4j import GraphDatabase
        >>> driver = GraphDatabase.driver("bolt://localhost:7687", auth=("neo4j", "password"))
        >>> writer = MergeKGWriter(driver=driver)
        >>> # Use in SimpleKGPipeline
        >>> pipeline = SimpleKGPipeline(..., kg_writer=writer)
    """

    def __init__(
        self,
        driver: Driver,
        neo4j_database: str | None = None,
    ):
        """Initialize the MERGE-based KG writer.

        Args:
            driver: Neo4j driver instance
            neo4j_database: Optional database name
        """
        self.driver = driver
        self.neo4j_database = neo4j_database
        self.hasher = ContentHasher()
        logger.info("Initialized MergeKGWriter with SHA256 content-based uniqueness")

    def _compute_content_hash(self, node: Any) -> str:
        """Compute SHA256 content hash based on node type.

        Different node types use different hashing strategies:
        - Document nodes: Hash file content (ensures file changes are detected)
        - Chunk nodes: Hash text content (ensures text changes are detected)
        - Entity nodes: Hash name + properties (ensures entity uniqueness)

        Args:
            node: Node object with label and properties

        Returns:
            SHA256 hex digest (64 characters)
        """
        props = node.properties.copy() if node.properties else {}

        # Document nodes: Hash actual file content
        if node.label == "Document" and "path" in props:
            try:
                content_hash = self.hasher.hash_file(props["path"])
                logger.debug(f"Document hash (file): {content_hash[:8]}... for {props['path']}")
                return content_hash
            except Exception as e:
                logger.warning(f"Failed to hash file {props['path']}: {e}, using entity hash")
                # Fallback to entity hash if file read fails

        # Chunk nodes: Hash text content
        elif node.label == "Chunk" and "text" in props:
            content_hash = self.hasher.hash_text(props["text"])
            logger.debug(f"Chunk hash (text): {content_hash[:8]}... for text length {len(props['text'])}")
            return content_hash

        # Entity nodes (Author, Paper, Concept, etc.): Hash name + properties
        name = props.get("name", "") or getattr(node, "id", "unknown")
        content_hash = self.hasher.hash_entity(name, props)
        logger.debug(f"Entity hash (name+props): {content_hash[:8]}... for {name}")
        return content_hash

    def _create_constraints(self, labels: set[str]) -> None:
        """Create SHA256-based uniqueness constraints for entity labels.

        Uses content_hash property (SHA256) instead of name for true
        content-based uniqueness and data integrity.

        Args:
            labels: Set of entity labels to create constraints for
        """
        with self.driver.session(database=self.neo4j_database) as session:
            for label in labels:
                try:
                    # Create constraint on 'content_hash' (SHA256 of content)
                    # This ensures true idempotency based on actual content
                    session.run(
                        f"CREATE CONSTRAINT {label.lower()}_content_hash_unique IF NOT EXISTS "
                        f"FOR (n:{label}) REQUIRE n.content_hash IS UNIQUE"
                    )
                    logger.debug(f"Created SHA256 constraint: {label}.content_hash")
                except Exception as e:
                    logger.debug(f"Constraint for {label} may already exist: {e}")

    @validate_call
    async def run(
        self,
        graph: Neo4jGraph,
        lexical_graph_config: LexicalGraphConfig = LexicalGraphConfig(),
    ) -> KGWriterModel:
        """Write knowledge graph using MERGE operations.

        This method writes entities and relationships to Neo4j using MERGE
        instead of CREATE, ensuring idempotency across multiple runs.

        Args:
            graph: Neo4jGraph containing nodes and relationships to write
            lexical_graph_config: Configuration for lexical graph structure

        Returns:
            KGWriterModel with write statistics
        """
        try:
            # Collect all unique labels from nodes
            labels = {node.label for node in graph.nodes}

            # Create uniqueness constraints before writing
            self._create_constraints(labels)

            # Write nodes using MERGE
            node_count = self._write_nodes_with_merge(graph)

            # Write relationships using MERGE
            rel_count = self._write_relationships_with_merge(graph)

            logger.info(
                f"MergeKGWriter: {node_count} nodes, {rel_count} relationships (MERGE operations)"
            )

            return KGWriterModel(
                status="SUCCESS",
                metadata={
                    "node_count": node_count,
                    "relationship_count": rel_count,
                },
            )
        except Exception as e:
            logger.error(f"MergeKGWriter failed: {e}")
            return KGWriterModel(
                status="FAILURE",
                metadata={"error": str(e)},
            )

    def _write_nodes_with_merge(self, graph: Neo4jGraph) -> int:
        """Write nodes using MERGE with SHA256 content hashing.

        Uses node-type-specific hashing strategies:
        - Document: SHA256 of file content
        - Chunk: SHA256 of text content
        - Entity: SHA256 of name + properties

        Args:
            graph: Neo4jGraph containing nodes

        Returns:
            Number of nodes processed
        """
        with self.driver.session(database=self.neo4j_database) as session:
            for node in graph.nodes:
                # Build properties dict
                props = node.properties.copy() if node.properties else {}

                # Compute SHA256 content hash based on node type
                # This uses appropriate hashing strategy for each node type
                content_hash = self._compute_content_hash(node)

                # MERGE on content_hash (matches unique constraint)
                merge_key = "content_hash: $content_hash"
                merge_props = {"content_hash": content_hash}

                # All original properties go into SET clause
                # This updates properties if content is reprocessed
                set_props = props.copy()
                set_props["content_hash"] = content_hash  # Store hash for reference

                # Build MERGE query
                query = f"MERGE (n:{node.label} {{{merge_key}}})"

                # Set all properties on CREATE and MATCH
                if set_props:
                    set_clause = ", ".join(f"n.{k} = ${k}" for k in set_props.keys())
                    query += f"\nON CREATE SET {set_clause}"
                    query += f"\nON MATCH SET {set_clause}"
                    merge_props.update(set_props)

                # Add __Entity__ label for entity resolution
                query += "\nSET n:__Entity__"

                # Add embedding if present
                if node.embedding_properties:
                    for emb_key, emb_value in node.embedding_properties.items():
                        query += f"\nWITH n\nCALL db.create.setNodeVectorProperty(n, '{emb_key}', $emb_{emb_key})"
                        merge_props[f"emb_{emb_key}"] = emb_value

                logger.debug(f"MERGE {node.label} with hash: {content_hash[:8]}...")
                session.run(query, **merge_props)

        return len(graph.nodes)

    def _write_relationships_with_merge(self, graph: Neo4jGraph) -> int:
        """Write relationships using pure Cypher MERGE with SHA256 content hashing.

        Each relationship gets a content_hash property for data integrity tracking.

        Args:
            graph: Neo4jGraph containing relationships

        Returns:
            Number of relationships processed
        """
        with self.driver.session(database=self.neo4j_database) as session:
            for rel in graph.relationships:
                # Get start and end nodes
                start_node = next((n for n in graph.nodes if n.id == rel.start_node_id), None)
                end_node = next((n for n in graph.nodes if n.id == rel.end_node_id), None)

                if not start_node or not end_node:
                    logger.warning(f"Skipping relationship {rel.type}: missing start or end node")
                    continue

                # Compute content hashes for both nodes
                start_hash = self._compute_content_hash(start_node)
                end_hash = self._compute_content_hash(end_node)

                # Compute relationship content hash
                rel_props = rel.properties if rel.properties else {}
                content_hash = self.hasher.hash_relationship(
                    rel.type, start_hash, end_hash, rel_props
                )
                logger.debug(
                    f"Relationship hash: {content_hash[:8]}... for {rel.type} "
                    f"({start_hash[:8]}...)-[{rel.type}]->({end_hash[:8]}...)"
                )

                # Build MATCH patterns with proper prefixes
                start_match = self._build_node_match("start", start_node)
                end_match = self._build_node_match("end", end_node)

                # Build relationship properties (include original props + content_hash)
                props_str = ""
                params = {}

                if rel.properties:
                    props_str = " {" + ", ".join(f"{k}: $rel_{k}" for k in rel.properties.keys()) + "}"
                    params = {f"rel_{k}": v for k, v in rel.properties.items()}

                # MERGE relationship using pure Cypher (no APOC)
                query = f"""
                MATCH {start_match}
                MATCH {end_match}
                MERGE (start)-[r:{rel.type}{props_str}]->(end)
                SET r.content_hash = $rel_content_hash
                RETURN r
                """

                # Add content_hash to params
                params["rel_content_hash"] = content_hash

                # Add start/end node matching properties
                params.update(self._get_node_params("start", start_node))
                params.update(self._get_node_params("end", end_node))

                session.run(query, **params)

        return len(graph.relationships)

    def _build_node_match(self, prefix: str, node: Any) -> str:
        """Build MATCH pattern for a node using SHA256 content hash.

        Args:
            prefix: "start" or "end" - determines variable name in query
            node: Node object

        Returns:
            Cypher MATCH pattern string (using content_hash)
        """
        label = node.label

        # Always match on content_hash (SHA256 of content)
        # This matches the MERGE key used in _write_nodes_with_merge
        return f"({prefix}:{label} {{content_hash: ${prefix}_content_hash}})"

    def _get_node_params(self, prefix: str, node: Any) -> dict[str, Any]:
        """Get parameters for node matching using SHA256 hash.

        Uses node-type-specific hashing (same as _write_nodes_with_merge).

        Args:
            prefix: Parameter prefix ("start" or "end")
            node: Node object

        Returns:
            Dictionary of parameters with content_hash
        """
        # Compute same content hash as used in _write_nodes_with_merge
        # This ensures consistent hashing across write and match operations
        content_hash = self._compute_content_hash(node)

        return {f"{prefix}_content_hash": content_hash}
