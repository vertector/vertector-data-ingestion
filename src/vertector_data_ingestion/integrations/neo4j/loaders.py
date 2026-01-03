"""Data loaders for Neo4j SimpleKGPipeline integration.

This module provides DataLoader implementations that integrate Vertector's multimodal
ingestion capabilities with Neo4j's SimpleKGPipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vertector_data_ingestion import (
    AudioConfig,
    ExportFormat,
    LocalMpsConfig,
    UniversalConverter,
    WhisperModelSize,
    create_audio_transcriber,
)
from vertector_data_ingestion.models.config import ConverterConfig
from vertector_data_ingestion.models.document import DoclingDocumentWrapper

try:
    from neo4j_graphrag.experimental.components.pdf_loader import (
        DataLoader,
        DocumentInfo,
        PdfDocument,
    )
except ImportError as e:
    msg = (
        "neo4j-graphrag is required for Neo4j integration. "
        "Install with: uv pip install vertector-data-ingestion[neo4j]"
    )
    raise ImportError(msg) from e


class VertectorDataLoader(DataLoader):
    """Custom data loader using Vertector for multimodal document ingestion.

    Supports: PDF, DOCX, PPTX, XLSX, HTML, Markdown, and more.

    This loader preserves the DoclingDocumentWrapper for downstream processing
    (e.g., by VertectorTextSplitter for rich metadata extraction).

    Args:
        config: Vertector configuration (LocalMpsConfig, CloudGpuConfig, CloudCpuConfig,
                or ConverterConfig). Defaults to LocalMpsConfig for hardware optimization.

    Attributes:
        converter: UniversalConverter instance
        last_document: Last processed DoclingDocumentWrapper (for rich chunking)
        last_metadata: Metadata from the last processed document

    Example:
        >>> from vertector_data_ingestion import LocalMpsConfig
        >>> from vertector_data_ingestion.integrations.neo4j import VertectorDataLoader
        >>>
        >>> loader = VertectorDataLoader(LocalMpsConfig())
        >>> result = await loader.run(Path("research_paper.pdf"))
        >>> print(result.text[:100])
        '# Document Title\\n\\nFirst paragraph...'
        >>> # Access rich document for chunking
        >>> doc_wrapper = loader.last_document
    """

    def __init__(self, config: ConverterConfig | None = None) -> None:
        """Initialize loader with Vertector configuration.

        Args:
            config: Configuration for document processing. If None, uses LocalMpsConfig
                   for Apple Silicon optimization.
        """
        self.converter = UniversalConverter(config or LocalMpsConfig())
        self.last_document: DoclingDocumentWrapper | None = None
        self.last_metadata: dict[str, Any] = {}

    async def run(self, filepath: Path, metadata: dict[str, str] | None = None) -> PdfDocument:
        """Load and convert document using Vertector.

        This method preserves the DoclingDocumentWrapper in `last_document` attribute
        for downstream processors (like VertectorTextSplitter) to access rich structure.

        Args:
            filepath: Path to document file (PDF, DOCX, PPTX, XLSX, etc.)
            metadata: Optional metadata to associate with the document

        Returns:
            PdfDocument with markdown text and metadata

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported

        Example:
            >>> loader = VertectorDataLoader()
            >>> result = await loader.run(Path("document.pdf"))
            >>> print(result.text[:100])
            '# Document Title\\n\\nFirst paragraph...'
        """
        # Convert using Vertector's unified API
        doc_wrapper = self.converter.convert(filepath)

        # Store for downstream access (enables rich chunking)
        self.last_document = doc_wrapper

        # Export to markdown for Neo4j text processing
        text = self.converter.export(doc_wrapper, ExportFormat.MARKDOWN)

        # Build enriched metadata
        doc_metadata = {
            "filename": filepath.name,
            "num_pages": str(doc_wrapper.metadata.num_pages),
            "pipeline_type": doc_wrapper.metadata.pipeline_type,
            "processing_time": str(doc_wrapper.metadata.processing_time),
            "file_size_bytes": str(filepath.stat().st_size),
        }

        # Merge with provided metadata
        if metadata:
            doc_metadata.update(metadata)

        # Store for external access
        self.last_metadata = doc_metadata

        # Create document info
        doc_info = DocumentInfo(
            path=str(filepath),
            metadata=doc_metadata,
            document_type="document",
        )

        return PdfDocument(text=text, document_info=doc_info)


class VertectorAudioLoader(DataLoader):
    """Audio loader using Vertector's Whisper transcription.

    Supports: WAV, MP3, M4A, FLAC, OGG audio formats.

    Args:
        config: AudioConfig for transcription settings. If None, uses BASE model
                with word timestamps enabled.

    Attributes:
        transcriber: Audio transcriber instance
        last_metadata: Metadata from the last transcribed audio file

    Example:
        >>> from vertector_data_ingestion import AudioConfig, WhisperModelSize
        >>> from vertector_data_ingestion.integrations.neo4j import VertectorAudioLoader
        >>>
        >>> config = AudioConfig(
        ...     model_size=WhisperModelSize.BASE,
        ...     word_timestamps=True
        ... )
        >>> loader = VertectorAudioLoader(config)
        >>> result = await loader.run(Path("meeting.wav"))
        >>> print(result.document_info.metadata["duration"])
        '180.5'
    """

    def __init__(self, config: AudioConfig | None = None) -> None:
        """Initialize audio transcriber.

        Args:
            config: Audio transcription configuration. If None, uses BASE Whisper model
                   with word-level timestamps.
        """
        if config is None:
            config = AudioConfig(
                model_size=WhisperModelSize.BASE,
                word_timestamps=True,
            )
        self.transcriber = create_audio_transcriber(config)
        self.last_transcription_result = None  # Store for splitter access
        self.last_metadata: dict[str, Any] = {}

    async def run(self, filepath: Path, metadata: dict[str, str] | None = None) -> PdfDocument:
        """Load and transcribe audio file.

        Args:
            filepath: Path to audio file (.wav, .mp3, .m4a, .flac, .ogg)
            metadata: Optional metadata to associate with the document

        Returns:
            PdfDocument with transcribed text and metadata

        Raises:
            FileNotFoundError: If the audio file does not exist
            ValueError: If the audio format is not supported

        Example:
            >>> loader = VertectorAudioLoader()
            >>> result = await loader.run(Path("interview.mp3"))
            >>> print(result.text)
            '# Audio Transcription: interview.mp3\\n**Duration:** 45.2s\\n...'
        """
        # Transcribe audio
        result = self.transcriber.transcribe(filepath)

        # Store for splitter access (enables audio segment chunking)
        self.last_transcription_result = result

        # Format as markdown with timestamps
        markdown_parts = [f"# Audio Transcription: {filepath.name}\n"]
        markdown_parts.append(f"**Duration:** {result.duration:.2f}s\n")
        markdown_parts.append(f"**Language:** {result.language}\n\n")

        for i, segment in enumerate(result.segments, 1):
            timestamp = f"[{segment.start:.1f}s - {segment.end:.1f}s]"
            markdown_parts.append(f"**{i}.** {timestamp} {segment.text}\n")

        text = "\n".join(markdown_parts)

        # Build enriched metadata
        audio_metadata = {
            "filename": filepath.name,
            "duration": str(result.duration),
            "language": result.language,
            "segments": str(len(result.segments)),
            "model": result.model_name,
            "modality": "audio",
        }

        # Merge with provided metadata
        if metadata:
            audio_metadata.update(metadata)

        # Store for external access
        self.last_metadata = audio_metadata

        # Create document info
        doc_info = DocumentInfo(
            path=str(filepath),
            metadata=audio_metadata,
            document_type="audio",
        )

        return PdfDocument(text=text, document_info=doc_info)


class MultimodalLoader(DataLoader):
    """Unified loader for documents and audio files.

    Automatically detects file type and delegates to appropriate loader:
    - Documents (PDF, DOCX, etc.): VertectorDataLoader
    - Audio (WAV, MP3, etc.): VertectorAudioLoader

    This class follows composition over inheritance, reusing the specialized
    loaders instead of duplicating code.

    Args:
        vertector_config: Configuration for document processing
        audio_config: Configuration for audio transcription

    Attributes:
        doc_loader: VertectorDataLoader instance for documents
        audio_loader: VertectorAudioLoader instance for audio

    Example:
        >>> from vertector_data_ingestion import LocalMpsConfig, AudioConfig
        >>> from vertector_data_ingestion.integrations.neo4j import MultimodalLoader
        >>>
        >>> loader = MultimodalLoader(
        ...     vertector_config=LocalMpsConfig(),
        ...     audio_config=AudioConfig()
        ... )
        >>>
        >>> # Process document
        >>> doc_result = await loader.run(Path("paper.pdf"))
        >>> # Access rich document for chunking
        >>> doc_wrapper = loader.doc_loader.last_document
        >>>
        >>> # Process audio
        >>> audio_result = await loader.run(Path("meeting.wav"))
    """

    # Audio file extensions supported by Whisper
    AUDIO_EXTENSIONS = {".wav", ".mp3", ".m4a", ".flac", ".ogg"}

    def __init__(
        self,
        vertector_config: ConverterConfig | None = None,
        audio_config: AudioConfig | None = None,
    ) -> None:
        """Initialize multimodal loader with specialized loaders.

        Args:
            vertector_config: Config for documents (LocalMpsConfig, CloudGpuConfig, etc.)
                             If None, uses LocalMpsConfig.
            audio_config: AudioConfig for audio files. If None, uses BASE Whisper model.
        """
        # Compose specialized loaders (DRY principle)
        self.doc_loader = VertectorDataLoader(config=vertector_config)
        self.audio_loader = VertectorAudioLoader(config=audio_config)

    @property
    def last_document(self) -> DoclingDocumentWrapper | None:
        """Get last processed document (for downstream rich chunking).

        Returns:
            DoclingDocumentWrapper if a document was processed, None otherwise
        """
        return self.doc_loader.last_document

    @property
    def last_metadata(self) -> dict[str, Any]:
        """Get metadata from the last used loader.

        Returns:
            Metadata dictionary from the most recently used loader
        """
        # Return metadata from whichever loader was used last
        # Check audio_loader first as it has 'modality' key
        if self.audio_loader.last_metadata.get("modality") == "audio":
            return self.audio_loader.last_metadata
        return self.doc_loader.last_metadata

    async def run(self, filepath: Path, metadata: dict[str, str] | None = None) -> PdfDocument:
        """Load any supported file type (document or audio).

        Automatically detects file type based on extension and delegates to
        the appropriate specialized loader.

        Args:
            filepath: Path to file (document or audio)
            metadata: Optional metadata to associate with the document

        Returns:
            PdfDocument with text and metadata

        Raises:
            FileNotFoundError: If the file does not exist
            ValueError: If the file format is not supported

        Example:
            >>> loader = MultimodalLoader()
            >>> doc = await loader.run(Path("report.pdf"))  # Uses doc_loader
            >>> audio = await loader.run(Path("call.mp3"))  # Uses audio_loader
        """
        # Detect file type by extension and delegate to specialized loader
        if filepath.suffix.lower() in self.AUDIO_EXTENSIONS:
            return await self.audio_loader.run(filepath, metadata)
        return await self.doc_loader.run(filepath, metadata)
