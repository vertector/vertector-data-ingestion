"""ChromaDB vector store adapter."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from vertector_data_ingestion.models.chunk import DocumentChunk
from vertector_data_ingestion.vector.base import VectorStoreAdapter


class ChromaAdapter(VectorStoreAdapter):
    """ChromaDB adapter for vector storage."""

    def __init__(
        self,
        collection_name: str = "documents",
        persist_directory: Optional[Path] = None,
        embedding_model: str = "Qwen/Qwen3-Embedding-4B",
    ):
        """
        Initialize ChromaDB adapter.

        Args:
            collection_name: Name of the collection
            persist_directory: Directory for persistent storage
            embedding_model: Sentence transformer model for embeddings
        """
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "ChromaDB and sentence-transformers required. "
                "Install with: uv add chromadb sentence-transformers"
            )

        self.collection_name = collection_name
        self.embedding_model_name = embedding_model

        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)

        # Initialize ChromaDB client
        if persist_directory:
            persist_directory = Path(persist_directory)
            persist_directory.mkdir(parents=True, exist_ok=True)
            self.client = chromadb.PersistentClient(path=str(persist_directory))
            logger.info(f"ChromaDB initialized with persistence at {persist_directory}")
        else:
            self.client = chromadb.Client()
            logger.info("ChromaDB initialized in-memory")

        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"embedding_model": embedding_model},
        )

        logger.info(f"Using collection: {collection_name}")

    def add_chunks(
        self, chunks: List[DocumentChunk], batch_size: int = 16
    ) -> None:
        """
        Add document chunks to ChromaDB in batches to avoid memory issues.

        Args:
            chunks: List of document chunks
            batch_size: Number of chunks to process at once (default: 16)
        """
        if not chunks:
            logger.warning("No chunks to add")
            return

        logger.info(f"Adding {len(chunks)} chunks to ChromaDB (batch_size={batch_size})")

        # Process in batches to avoid memory issues
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]

            logger.debug(
                f"Processing batch {batch_idx + 1}/{total_batches} "
                f"(chunks {start_idx + 1}-{end_idx})"
            )

            # Extract texts for embedding
            texts = [chunk.text for chunk in batch_chunks]

            # Generate embeddings for this batch
            embeddings = self.embedding_model.encode(
                texts,
                show_progress_bar=False,  # Disable per-batch progress
                batch_size=batch_size,  # Internal batch size for encoding
            )

            # Prepare data for ChromaDB
            ids = [chunk.chunk_id for chunk in batch_chunks]
            metadatas = []

            for chunk in batch_chunks:
                metadata = chunk.to_dict()

                # Remove embedding from metadata if present (avoid duplication)
                metadata.pop("embedding", None)

                # Convert Path objects to strings
                if "source_path" in metadata:
                    metadata["source_path"] = str(metadata["source_path"])

                # Flatten nested dicts (ChromaDB only supports flat metadata)
                clean_metadata = {}
                for key, value in metadata.items():
                    if value is None:
                        # Skip None values - ChromaDB doesn't accept them
                        continue
                    elif isinstance(value, (str, int, float, bool)):
                        clean_metadata[key] = value
                    elif isinstance(value, dict):
                        # Flatten nested dict by prefixing keys
                        for nested_key, nested_value in value.items():
                            if nested_value is None:
                                # Skip None values
                                continue
                            elif isinstance(nested_value, (str, int, float, bool)):
                                flat_key = f"{key}_{nested_key}"
                                clean_metadata[flat_key] = nested_value
                    else:
                        # Convert other types to string
                        clean_metadata[key] = str(value)

                metadatas.append(clean_metadata)

            # Add batch to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
            )

            logger.debug(f"Added batch {batch_idx + 1}/{total_batches}")

        logger.info(f"Successfully added {len(chunks)} chunks in {total_batches} batches")

    def search(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query: Query text
            top_k: Number of results to return
            filters: Optional metadata filters (e.g., {"source_path": "doc.pdf"})

        Returns:
            List of search results with chunk data and similarity scores
        """
        logger.debug(f"Searching for: {query} (top_k={top_k})")

        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]

        # Build where clause for filters
        where = None
        if filters:
            where = filters

        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
        )

        # Format results
        search_results = []
        for i in range(len(results["ids"][0])):
            distance = results["distances"][0][i] if "distances" in results else None
            # Convert distance to similarity score (smaller distance = higher similarity)
            score = 1.0 - distance if distance is not None else None

            search_results.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": distance,
                    "score": score,
                }
            )

        logger.debug(f"Found {len(search_results)} results")

        return search_results

    def delete_by_source(self, source_path: str) -> None:
        """
        Delete all chunks from a specific source document.

        Args:
            source_path: Source document path
        """
        logger.info(f"Deleting chunks from: {source_path}")

        # Query for chunks from this source
        results = self.collection.get(where={"source_path": source_path})

        if results["ids"]:
            # Delete chunks
            self.collection.delete(ids=results["ids"])
            logger.info(f"Deleted {len(results['ids'])} chunks")
        else:
            logger.info("No chunks found to delete")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get ChromaDB collection statistics.

        Returns:
            Dictionary with collection stats
        """
        count = self.collection.count()

        return {
            "collection_name": self.collection_name,
            "total_chunks": count,
            "embedding_model": self.embedding_model_name,
        }

    def clear(self) -> None:
        """Delete all chunks from collection."""
        logger.warning(f"Clearing collection: {self.collection_name}")
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"embedding_model": self.embedding_model_name},
        )
        logger.info("Collection cleared")
