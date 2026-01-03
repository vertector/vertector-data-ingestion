"""Unit tests for Neo4j loaders using mocks."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from vertector_data_ingestion import AudioConfig, LocalMpsConfig, WhisperModelSize


@pytest.mark.asyncio
class TestVertectorDataLoader:
    """Test suite for VertectorDataLoader."""

    @pytest.fixture
    def mock_doc_wrapper(self):
        """Create mock DoclingDocumentWrapper."""
        mock = MagicMock()
        mock.metadata.num_pages = 10
        mock.metadata.pipeline_type = "standard"
        mock.metadata.processing_time = 1.5
        mock.metadata.source_path = Path("test.pdf")
        return mock

    @pytest.fixture
    def mock_converter(self, mock_doc_wrapper):
        """Create mock UniversalConverter."""
        with patch("vertector_data_ingestion.integrations.neo4j.loaders.UniversalConverter") as mock:
            instance = mock.return_value
            instance.convert.return_value = mock_doc_wrapper
            instance.export.return_value = "# Test Document\n\nTest content"
            yield instance

    async def test_loader_initialization(self):
        """Test loader initializes with default config."""
        from vertector_data_ingestion.integrations.neo4j import VertectorDataLoader

        loader = VertectorDataLoader()
        assert loader.last_document is None
        assert loader.last_metadata == {}

    async def test_loader_with_custom_config(self):
        """Test loader initialization with custom config."""
        from vertector_data_ingestion.integrations.neo4j import VertectorDataLoader

        config = LocalMpsConfig()
        loader = VertectorDataLoader(config=config)
        assert loader.converter is not None

    async def test_run_converts_document(self, mock_converter, mock_doc_wrapper):
        """Test run() converts document and stores metadata."""
        from vertector_data_ingestion.integrations.neo4j import VertectorDataLoader

        with patch("vertector_data_ingestion.integrations.neo4j.loaders.UniversalConverter") as MockConverter:
            MockConverter.return_value = mock_converter
            loader = VertectorDataLoader()
            result = await loader.run(Path("test.pdf"))

            # Verify converter was called
            mock_converter.convert.assert_called_once()
            mock_converter.export.assert_called_once()

            # Verify result
            assert result.text == "# Test Document\n\nTest content"
            assert result.document_info.path == "test.pdf"
            assert result.document_info.document_type == "document"

            # Verify metadata
            assert loader.last_metadata["filename"] == "test.pdf"
            assert loader.last_metadata["num_pages"] == "10"

            # Verify document wrapper is stored
            assert loader.last_document == mock_doc_wrapper

    async def test_run_with_metadata(self, mock_converter, mock_doc_wrapper):
        """Test run() merges provided metadata."""
        from vertector_data_ingestion.integrations.neo4j import VertectorDataLoader

        with patch("vertector_data_ingestion.integrations.neo4j.loaders.UniversalConverter") as MockConverter:
            MockConverter.return_value = mock_converter
            loader = VertectorDataLoader()
            custom_meta = {"author": "Test Author", "category": "research"}
            result = await loader.run(Path("test.pdf"), metadata=custom_meta)

            # Verify custom metadata is included
            assert result.document_info.metadata["author"] == "Test Author"
            assert result.document_info.metadata["category"] == "research"


@pytest.mark.asyncio
class TestVertectorAudioLoader:
    """Test suite for VertectorAudioLoader."""

    @pytest.fixture
    def mock_transcription_result(self):
        """Create mock TranscriptionResult."""
        mock = MagicMock()
        mock.duration = 120.5
        mock.language = "en"
        mock.model_name = "mlx-whisper-base"

        # Mock segments
        seg1 = MagicMock()
        seg1.start = 0.0
        seg1.end = 5.5
        seg1.text = "This is the first segment."

        seg2 = MagicMock()
        seg2.start = 5.5
        seg2.end = 10.2
        seg2.text = "This is the second segment."

        mock.segments = [seg1, seg2]
        return mock

    async def test_loader_initialization_default_config(self):
        """Test loader initializes with default audio config."""
        from vertector_data_ingestion.integrations.neo4j import VertectorAudioLoader

        with patch("vertector_data_ingestion.integrations.neo4j.loaders.create_audio_transcriber"):
            loader = VertectorAudioLoader()
            assert loader.last_metadata == {}

    async def test_loader_with_custom_config(self):
        """Test loader initialization with custom audio config."""
        from vertector_data_ingestion.integrations.neo4j import VertectorAudioLoader

        config = AudioConfig(model_size=WhisperModelSize.SMALL)
        with patch("vertector_data_ingestion.integrations.neo4j.loaders.create_audio_transcriber") as mock_create:
            loader = VertectorAudioLoader(config=config)
            mock_create.assert_called_once_with(config)

    async def test_run_transcribes_audio(self, mock_transcription_result):
        """Test run() transcribes audio and formats output."""
        from vertector_data_ingestion.integrations.neo4j import VertectorAudioLoader

        with patch("vertector_data_ingestion.integrations.neo4j.loaders.create_audio_transcriber") as mock_create:
            mock_transcriber = MagicMock()
            mock_transcriber.transcribe.return_value = mock_transcription_result
            mock_create.return_value = mock_transcriber

            loader = VertectorAudioLoader()
            result = await loader.run(Path("test.wav"))

            # Verify transcriber was called
            mock_transcriber.transcribe.assert_called_once()

            # Verify result
            assert "Audio Transcription: test.wav" in result.text
            assert "Duration: 120.50s" in result.text
            assert "Language: en" in result.text
            assert "first segment" in result.text
            assert "second segment" in result.text

            # Verify metadata
            assert result.document_info.document_type == "audio"
            assert loader.last_metadata["duration"] == "120.5"
            assert loader.last_metadata["language"] == "en"
            assert loader.last_metadata["segments"] == "2"
            assert loader.last_metadata["modality"] == "audio"


@pytest.mark.asyncio
class TestMultimodalLoader:
    """Test suite for MultimodalLoader."""

    async def test_loader_initialization(self):
        """Test multimodal loader initializes both loaders."""
        from vertector_data_ingestion.integrations.neo4j import MultimodalLoader

        with patch("vertector_data_ingestion.integrations.neo4j.loaders.VertectorDataLoader"), \
             patch("vertector_data_ingestion.integrations.neo4j.loaders.VertectorAudioLoader"):
            loader = MultimodalLoader()
            assert loader.doc_loader is not None
            assert loader.audio_loader is not None

    async def test_routes_to_audio_loader(self):
        """Test multimodal loader routes audio files to audio loader."""
        from vertector_data_ingestion.integrations.neo4j import MultimodalLoader

        with patch("vertector_data_ingestion.integrations.neo4j.loaders.VertectorDataLoader"), \
             patch("vertector_data_ingestion.integrations.neo4j.loaders.VertectorAudioLoader"):
            loader = MultimodalLoader()
            loader.audio_loader.run = AsyncMock(return_value=MagicMock())

            await loader.run(Path("test.mp3"))
            loader.audio_loader.run.assert_called_once()

    async def test_routes_to_doc_loader(self):
        """Test multimodal loader routes documents to document loader."""
        from vertector_data_ingestion.integrations.neo4j import MultimodalLoader

        with patch("vertector_data_ingestion.integrations.neo4j.loaders.VertectorDataLoader"), \
             patch("vertector_data_ingestion.integrations.neo4j.loaders.VertectorAudioLoader"):
            loader = MultimodalLoader()
            loader.doc_loader.run = AsyncMock(return_value=MagicMock())

            await loader.run(Path("test.pdf"))
            loader.doc_loader.run.assert_called_once()

    async def test_last_metadata_property_audio(self):
        """Test last_metadata returns audio metadata when audio was processed."""
        from vertector_data_ingestion.integrations.neo4j import MultimodalLoader

        with patch("vertector_data_ingestion.integrations.neo4j.loaders.VertectorDataLoader"), \
             patch("vertector_data_ingestion.integrations.neo4j.loaders.VertectorAudioLoader"):
            loader = MultimodalLoader()
            loader.audio_loader.last_metadata = {"modality": "audio", "duration": "120"}
            loader.doc_loader.last_metadata = {"num_pages": "10"}

            assert loader.last_metadata["modality"] == "audio"

    async def test_last_metadata_property_document(self):
        """Test last_metadata returns document metadata when document was processed."""
        from vertector_data_ingestion.integrations.neo4j import MultimodalLoader

        with patch("vertector_data_ingestion.integrations.neo4j.loaders.VertectorDataLoader"), \
             patch("vertector_data_ingestion.integrations.neo4j.loaders.VertectorAudioLoader"):
            loader = MultimodalLoader()
            loader.audio_loader.last_metadata = {}
            loader.doc_loader.last_metadata = {"num_pages": "10", "filename": "test.pdf"}

            assert loader.last_metadata["num_pages"] == "10"

    async def test_last_document_property(self):
        """Test last_document property accesses doc_loader.last_document."""
        from vertector_data_ingestion.integrations.neo4j import MultimodalLoader

        with patch("vertector_data_ingestion.integrations.neo4j.loaders.VertectorDataLoader"), \
             patch("vertector_data_ingestion.integrations.neo4j.loaders.VertectorAudioLoader"):
            loader = MultimodalLoader()
            mock_doc = MagicMock()
            loader.doc_loader.last_document = mock_doc

            assert loader.last_document == mock_doc
