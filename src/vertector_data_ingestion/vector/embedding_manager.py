"""Embedding model management."""

from loguru import logger
from sentence_transformers import SentenceTransformer


class EmbeddingManager:
    """Manager for embedding models."""

    def __init__(self, model_name: str = "Qwen/Qwen3-Embedding-4B"):
        """
        Initialize embedding manager.

        Args:
            model_name: Name of sentence transformer model
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)

        logger.info(
            f"Embedding model loaded. Dimension: {self.model.get_sentence_embedding_dimension()}"
        )

    def encode(
        self,
        texts: list[str],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True,
    ) -> list[list[float]]:
        """
        Encode texts to embeddings.

        Args:
            texts: List of texts to encode
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: Normalize embeddings to unit length

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
        )

        return embeddings.tolist()

    def encode_single(self, text: str, normalize: bool = True) -> list[float]:
        """
        Encode single text to embedding.

        Args:
            text: Text to encode
            normalize: Normalize embedding

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            [text], normalize_embeddings=normalize, show_progress_bar=False
        )[0]

        return embedding.tolist()

    def get_dimension(self) -> int:
        """
        Get embedding dimension.

        Returns:
            Embedding vector dimension
        """
        return self.model.get_sentence_embedding_dimension()

    @staticmethod
    def get_recommended_models() -> dict:
        """
        Get recommended embedding models for different use cases.

        Returns:
            Dictionary of use case -> model name mappings
        """
        return {
            "fast_small": "sentence-transformers/all-MiniLM-L6-v2",  # 384 dim, fast
            "balanced": "sentence-transformers/all-mpnet-base-v2",  # 768 dim, good quality
            "high_quality": "sentence-transformers/all-MiniLM-L12-v2",  # 384 dim, better quality
            "multilingual": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
            "code": "sentence-transformers/all-MiniLM-L6-v2",  # Also works for code
        }
