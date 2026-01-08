"""SHA256-based content hashing for data integrity and idempotency.

This module provides utilities for generating cryptographic hashes of content
to ensure true idempotency and data integrity in the knowledge graph.

Key Features:
- Document content hashing (file-based SHA256)
- Chunk text hashing (content-based SHA256)
- Entity composite hashing (name + properties)
- Relationship composite hashing (type + nodes + properties)

Example:
    >>> hasher = ContentHasher()
    >>> doc_hash = hasher.hash_file("document.pdf")
    >>> chunk_hash = hasher.hash_text("This is chunk text")
    >>> entity_hash = hasher.hash_entity("John Smith", {"age": 30})
"""

import hashlib
import json
from pathlib import Path
from typing import Any


class ContentHasher:
    """Utility class for generating SHA256 content hashes."""

    @staticmethod
    def hash_file(file_path: str | Path) -> str:
        """Generate SHA256 hash of file contents.

        Args:
            file_path: Path to file

        Returns:
            SHA256 hex digest (64 characters)

        Example:
            >>> hasher = ContentHasher()
            >>> hash1 = hasher.hash_file("doc.pdf")
            >>> hash2 = hasher.hash_file("doc.pdf")
            >>> hash1 == hash2  # True - same file = same hash
        """
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            # Read in 64kb chunks for memory efficiency
            while chunk := f.read(65536):
                sha256.update(chunk)

        return sha256.hexdigest()

    @staticmethod
    def hash_text(text: str) -> str:
        """Generate SHA256 hash of text content.

        Args:
            text: Text content to hash

        Returns:
            SHA256 hex digest (64 characters)

        Example:
            >>> hasher = ContentHasher()
            >>> hash1 = hasher.hash_text("Hello World")
            >>> hash2 = hasher.hash_text("Hello World")
            >>> hash1 == hash2  # True - same text = same hash
        """
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_entity(name: str, properties: dict[str, Any] | None = None) -> str:
        """Generate SHA256 hash for entity based on name + properties.

        This creates a composite hash that includes both the entity name
        and all its properties, ensuring entities with the same name but
        different properties get unique hashes.

        Args:
            name: Entity name
            properties: Entity properties (optional)

        Returns:
            SHA256 hex digest (64 characters)

        Example:
            >>> hasher = ContentHasher()
            >>> hash1 = hasher.hash_entity("John Smith", {"age": 30})
            >>> hash2 = hasher.hash_entity("John Smith", {"age": 40})
            >>> hash1 != hash2  # True - different properties = different hash
        """
        props = properties or {}

        # Create deterministic representation by sorting keys
        # This ensures consistent hashing regardless of dict order
        composite = {
            "name": name,
            "properties": {k: props[k] for k in sorted(props.keys())},
        }

        # Use JSON with sort_keys for deterministic serialization
        composite_str = json.dumps(composite, sort_keys=True, ensure_ascii=True)

        return hashlib.sha256(composite_str.encode("utf-8")).hexdigest()

    @staticmethod
    def hash_relationship(
        rel_type: str,
        start_node_hash: str,
        end_node_hash: str,
        properties: dict[str, Any] | None = None,
    ) -> str:
        """Generate SHA256 hash for relationship.

        Creates a composite hash based on relationship type, connected nodes,
        and properties. This ensures relationship uniqueness.

        Args:
            rel_type: Relationship type (e.g., "AUTHORED", "KNOWS")
            start_node_hash: Hash of start node
            end_node_hash: Hash of end node
            properties: Relationship properties (optional)

        Returns:
            SHA256 hex digest (64 characters)

        Example:
            >>> hasher = ContentHasher()
            >>> rel_hash = hasher.hash_relationship(
            ...     "KNOWS",
            ...     "abc123...",  # Start node hash
            ...     "def456...",  # End node hash
            ...     {"since": 2020}
            ... )
        """
        props = properties or {}

        composite = {
            "type": rel_type,
            "start": start_node_hash,
            "end": end_node_hash,
            "properties": {k: props[k] for k in sorted(props.keys())},
        }

        composite_str = json.dumps(composite, sort_keys=True, ensure_ascii=True)

        return hashlib.sha256(composite_str.encode("utf-8")).hexdigest()


# Convenience function for backward compatibility
def hash_content(content: str | bytes) -> str:
    """Generate SHA256 hash of content.

    Args:
        content: String or bytes to hash

    Returns:
        SHA256 hex digest
    """
    if isinstance(content, str):
        content = content.encode("utf-8")

    return hashlib.sha256(content).hexdigest()


__all__ = ["ContentHasher", "hash_content"]
