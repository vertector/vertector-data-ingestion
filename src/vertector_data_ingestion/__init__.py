"""Vertector Data Ingestion - Universal multimodal data ingestion pipeline."""

__version__ = "0.1.0"

from vertector_data_ingestion.audio import (
    AudioTranscriber,
    TranscriptionResult,
    TranscriptionSegment,
    create_audio_transcriber,
)
from vertector_data_ingestion.chunkers.hybrid_chunker import HybridChunker
from vertector_data_ingestion.core.hardware_detector import HardwareDetector
from vertector_data_ingestion.core.pipeline_router import PipelineRouter
from vertector_data_ingestion.core.universal_converter import UniversalConverter
from vertector_data_ingestion.models.chunk import ChunkingResult, DocumentChunk
from vertector_data_ingestion.models.config import (
    AudioBackend,
    AudioConfig,
    CloudCpuConfig,
    CloudGpuConfig,
    ConverterConfig,
    ExportFormat,
    LocalMpsConfig,
    PipelineType,
    WhisperModelSize,
)
from vertector_data_ingestion.models.document import DoclingDocumentWrapper
from vertector_data_ingestion.monitoring.logger import setup_logging
from vertector_data_ingestion.monitoring.metrics import MetricsCollector
from vertector_data_ingestion.vector.chroma_adapter import ChromaAdapter

__all__ = [
    # Core
    "UniversalConverter",
    "PipelineRouter",
    "HardwareDetector",
    # Config
    "ConverterConfig",
    "LocalMpsConfig",
    "CloudGpuConfig",
    "CloudCpuConfig",
    "PipelineType",
    "ExportFormat",
    "AudioConfig",
    "WhisperModelSize",
    "AudioBackend",
    # Models
    "DoclingDocumentWrapper",
    "DocumentChunk",
    "ChunkingResult",
    # RAG
    "HybridChunker",
    "ChromaAdapter",
    # Audio
    "create_audio_transcriber",
    "AudioTranscriber",
    "TranscriptionResult",
    "TranscriptionSegment",
    # Monitoring
    "setup_logging",
    "MetricsCollector",
]
