"""Base vector store interface."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional

from vertector_data_ingestion.models.chunk import DocumentChunk


class VectorStoreAdapter(ABC):
    """Abstract base class for vector store adapters."""

    @abstractmethod
    def add_chunks(self, chunks: List[DocumentChunk]) -> None:
        """
        Add document chunks to vector store.

        Args:
            chunks: List of document chunks with text and metadata
        """
        pass

    @abstractmethod
    def search(
        self, query: str, top_k: int = 5, filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar chunks.

        Args:
            query: Query text
            top_k: Number of results to return
            filters: Optional metadata filters

        Returns:
            List of search results with chunk data and scores
        """
        pass

    @abstractmethod
    def delete_by_source(self, source_path: str) -> None:
        """
        Delete all chunks from a specific source document.

        Args:
            source_path: Source document path
        """
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        """
        Get vector store statistics.

        Returns:
            Dictionary with store statistics
        """
        pass
