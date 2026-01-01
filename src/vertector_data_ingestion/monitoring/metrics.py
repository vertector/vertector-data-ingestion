"""Performance metrics tracking."""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loguru import logger


@dataclass
class ConversionMetrics:
    """Metrics for document conversion."""

    total_documents: int = 0
    successful: int = 0
    failed: int = 0
    total_pages: int = 0
    total_time_seconds: float = 0.0
    conversion_times: List[float] = field(default_factory=list)

    @property
    def pages_per_second(self) -> float:
        """Calculate pages per second throughput."""
        if self.total_time_seconds == 0:
            return 0.0
        return self.total_pages / self.total_time_seconds

    @property
    def avg_time_per_document(self) -> float:
        """Calculate average time per document."""
        if self.total_documents == 0:
            return 0.0
        return self.total_time_seconds / self.total_documents

    @property
    def success_rate(self) -> float:
        """Calculate success rate percentage."""
        if self.total_documents == 0:
            return 0.0
        return (self.successful / self.total_documents) * 100

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "total_documents": self.total_documents,
            "successful": self.successful,
            "failed": self.failed,
            "total_pages": self.total_pages,
            "total_time_seconds": round(self.total_time_seconds, 2),
            "pages_per_second": round(self.pages_per_second, 2),
            "avg_time_per_document": round(self.avg_time_per_document, 2),
            "success_rate": round(self.success_rate, 2),
        }


@dataclass
class ChunkingMetrics:
    """Metrics for document chunking."""

    total_documents: int = 0
    total_chunks: int = 0
    total_tokens: int = 0
    avg_chunk_size: float = 0.0
    chunking_time_seconds: float = 0.0

    @property
    def chunks_per_document(self) -> float:
        """Calculate average chunks per document."""
        if self.total_documents == 0:
            return 0.0
        return self.total_chunks / self.total_documents

    @property
    def tokens_per_chunk(self) -> float:
        """Calculate average tokens per chunk."""
        if self.total_chunks == 0:
            return 0.0
        return self.total_tokens / self.total_chunks

    def to_dict(self) -> Dict:
        """Convert metrics to dictionary."""
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "total_tokens": self.total_tokens,
            "avg_chunk_size": round(self.avg_chunk_size, 2),
            "chunks_per_document": round(self.chunks_per_document, 2),
            "tokens_per_chunk": round(self.tokens_per_chunk, 2),
            "chunking_time_seconds": round(self.chunking_time_seconds, 2),
        }


class MetricsCollector:
    """Collects and tracks pipeline metrics."""

    def __init__(self):
        """Initialize metrics collector."""
        self.conversion_metrics = ConversionMetrics()
        self.chunking_metrics = ChunkingMetrics()
        self.start_time = time.time()

    def record_conversion(
        self,
        success: bool,
        pages: int,
        processing_time: float,
    ) -> None:
        """
        Record a document conversion.

        Args:
            success: Whether conversion succeeded
            pages: Number of pages processed
            processing_time: Time taken in seconds
        """
        self.conversion_metrics.total_documents += 1
        if success:
            self.conversion_metrics.successful += 1
            self.conversion_metrics.total_pages += pages
        else:
            self.conversion_metrics.failed += 1

        self.conversion_metrics.total_time_seconds += processing_time
        self.conversion_metrics.conversion_times.append(processing_time)

        logger.debug(
            f"Recorded conversion: success={success}, pages={pages}, time={processing_time:.2f}s"
        )

    def record_chunking(
        self,
        num_chunks: int,
        total_tokens: int,
        avg_chunk_size: float,
        processing_time: float,
    ) -> None:
        """
        Record chunking metrics.

        Args:
            num_chunks: Number of chunks created
            total_tokens: Total tokens across all chunks
            avg_chunk_size: Average chunk size
            processing_time: Time taken in seconds
        """
        self.chunking_metrics.total_documents += 1
        self.chunking_metrics.total_chunks += num_chunks
        self.chunking_metrics.total_tokens += total_tokens
        self.chunking_metrics.chunking_time_seconds += processing_time

        # Update rolling average
        total_docs = self.chunking_metrics.total_documents
        self.chunking_metrics.avg_chunk_size = (
            self.chunking_metrics.avg_chunk_size * (total_docs - 1) + avg_chunk_size
        ) / total_docs

        logger.debug(
            f"Recorded chunking: chunks={num_chunks}, tokens={total_tokens}, time={processing_time:.2f}s"
        )

    def get_summary(self) -> Dict:
        """
        Get summary of all metrics.

        Returns:
            Dictionary with all metrics
        """
        elapsed_time = time.time() - self.start_time

        return {
            "elapsed_time_seconds": round(elapsed_time, 2),
            "conversion": self.conversion_metrics.to_dict(),
            "chunking": self.chunking_metrics.to_dict(),
        }

    def log_summary(self) -> None:
        """Log metrics summary."""
        summary = self.get_summary()

        logger.info("=" * 60)
        logger.info("PIPELINE METRICS SUMMARY")
        logger.info("=" * 60)

        logger.info(f"Total elapsed time: {summary['elapsed_time_seconds']}s")

        logger.info("\nConversion Metrics:")
        for key, value in summary["conversion"].items():
            logger.info(f"  {key}: {value}")

        if self.chunking_metrics.total_documents > 0:
            logger.info("\nChunking Metrics:")
            for key, value in summary["chunking"].items():
                logger.info(f"  {key}: {value}")

        logger.info("=" * 60)

    def reset(self) -> None:
        """Reset all metrics."""
        self.conversion_metrics = ConversionMetrics()
        self.chunking_metrics = ChunkingMetrics()
        self.start_time = time.time()
        logger.info("Metrics reset")
