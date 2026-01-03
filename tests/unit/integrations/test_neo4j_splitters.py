"""Unit tests for Neo4j splitters using mocks."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from vertector_data_ingestion.models.chunk import ChunkingResult, DocumentChunk


@pytest.mark.asyncio
class TestVertectorTextSplitter:
    """Test suite for VertectorTextSplitter."""

    @pytest.fixture
    def mock_loader(self):
        """Create mock VertectorDataLoader with last_document."""
        mock = MagicMock()
        mock_doc = MagicMock()
        mock_doc.metadata.source_path = Path("test.pdf")
        mock_doc.metadata.num_pages = 5
        mock.last_document = mock_doc
        return mock

    @pytest.fixture
    def mock_chunks(self):
        """Create mock DocumentChunk list."""
        chunk1 = DocumentChunk(
            chunk_id="test_0",
            text="First chunk text with important content.",
            token_count=100,
            source_path=Path("test.pdf"),
            document_id="test",
            chunk_index=0,
            page_no=1,
            section_title="Introduction",
            is_table=False,
            is_heading=False,
            bbox={"l": 0.0, "t": 0.0, "r": 100.0, "b": 50.0},
            subsection_path="Chapter 1 > Section 1.1",
        )

        chunk2 = DocumentChunk(
            chunk_id="test_1",
            text="Second chunk with table data.",
            token_count=80,
            source_path=Path("test.pdf"),
            document_id="test",
            chunk_index=1,
            page_no=2,
            section_title="Methods",
            is_table=True,
            is_heading=False,
            bbox={"l": 0.0, "t": 60.0, "r": 100.0, "b": 110.0},
            subsection_path="Chapter 2 > Table 1",
        )

        return [chunk1, chunk2]

    @pytest.fixture
    def mock_chunking_result(self, mock_chunks):
        """Create mock ChunkingResult."""
        return ChunkingResult(
            source_path=Path("test.pdf"),
            chunks=mock_chunks,
            total_chunks=2,
            total_tokens=180,
            chunking_strategy="hybrid",
            avg_chunk_size=90.0,
            min_chunk_size=80,
            max_chunk_size=100,
        )

    def test_splitter_initialization(self, mock_loader):
        """Test splitter initializes with loader reference."""
        from vertector_data_ingestion.integrations.neo4j import VertectorTextSplitter

        splitter = VertectorTextSplitter(loader=mock_loader, chunk_size=512)
        assert splitter.loader == mock_loader
        assert splitter.last_chunks == []

    def test_splitter_invalid_chunk_size(self, mock_loader):
        """Test splitter raises error for invalid chunk size."""
        from vertector_data_ingestion.integrations.neo4j import VertectorTextSplitter

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            VertectorTextSplitter(loader=mock_loader, chunk_size=0)

        with pytest.raises(ValueError, match="chunk_size must be positive"):
            VertectorTextSplitter(loader=mock_loader, chunk_size=-10)

    async def test_run_raises_without_document(self):
        """Test run() raises error when no document is loaded."""
        from vertector_data_ingestion.integrations.neo4j import VertectorTextSplitter

        mock_loader = MagicMock()
        mock_loader.last_document = None

        splitter = VertectorTextSplitter(loader=mock_loader)

        with pytest.raises(RuntimeError, match="No document available"):
            await splitter.run("some text")

    async def test_run_chunks_document(self, mock_loader, mock_chunking_result):
        """Test run() uses chunk_document() and converts to Neo4j format."""
        from vertector_data_ingestion.integrations.neo4j import VertectorTextSplitter

        splitter = VertectorTextSplitter(loader=mock_loader, chunk_size=512)
        splitter.chunker.chunk_document = MagicMock(return_value=mock_chunking_result)

        result = await splitter.run("input text")

        # Verify chunker was called with document
        splitter.chunker.chunk_document.assert_called_once_with(
            mock_loader.last_document, include_metadata=True
        )

        # Verify result structure
        assert len(result.chunks) == 2

        # Verify first chunk
        chunk0 = result.chunks[0]
        assert chunk0.text == "First chunk text with important content."
        assert chunk0.index == 0
        assert chunk0.metadata["page_no"] == "1"
        assert chunk0.metadata["section_title"] == "Introduction"
        assert chunk0.metadata["token_count"] == "100"
        assert chunk0.metadata["subsection_path"] == "Chapter 1 > Section 1.1"
        assert "bbox" in chunk0.metadata

        # Verify second chunk
        chunk1 = result.chunks[1]
        assert chunk1.text == "Second chunk with table data."
        assert chunk1.index == 1
        assert chunk1.metadata["is_table"] == "True"
        assert chunk1.metadata["page_no"] == "2"

    def test_extract_metadata(self, mock_loader, mock_chunks):
        """Test _extract_metadata() extracts all fields correctly."""
        from vertector_data_ingestion.integrations.neo4j import VertectorTextSplitter

        splitter = VertectorTextSplitter(loader=mock_loader)
        chunk = mock_chunks[0]

        metadata = splitter._extract_metadata(chunk)

        assert metadata["chunk_id"] == "test_0"
        assert metadata["token_count"] == "100"
        assert metadata["document_id"] == "test"
        assert metadata["page_no"] == "1"
        assert metadata["section_title"] == "Introduction"
        # is_table and is_heading are False by default, only added if True
        assert "is_table" not in metadata  # Not added when False
        assert "is_heading" not in metadata  # Not added when False
        assert metadata["subsection_path"] == "Chapter 1 > Section 1.1"
        assert metadata["bbox"] == "0.0,0.0,100.0,50.0"

    def test_extract_metadata_table_chunk(self, mock_loader, mock_chunks):
        """Test _extract_metadata() handles table chunks."""
        from vertector_data_ingestion.integrations.neo4j import VertectorTextSplitter

        splitter = VertectorTextSplitter(loader=mock_loader)
        chunk = mock_chunks[1]

        metadata = splitter._extract_metadata(chunk)

        assert metadata["is_table"] == "True"
        assert metadata["bbox"] == "0.0,60.0,100.0,110.0"

    def test_get_chunk_metadata(self, mock_loader, mock_chunks):
        """Test get_chunk_metadata() retrieves metadata by index."""
        from vertector_data_ingestion.integrations.neo4j import VertectorTextSplitter

        splitter = VertectorTextSplitter(loader=mock_loader)
        splitter.last_chunks = mock_chunks

        # Get first chunk metadata
        meta0 = splitter.get_chunk_metadata(0)
        assert meta0["page_no"] == "1"
        assert meta0["section_title"] == "Introduction"

        # Get second chunk metadata
        meta1 = splitter.get_chunk_metadata(1)
        assert meta1["page_no"] == "2"
        assert meta1["is_table"] == "True"

    def test_get_chunk_metadata_out_of_range(self, mock_loader, mock_chunks):
        """Test get_chunk_metadata() returns empty dict for invalid index."""
        from vertector_data_ingestion.integrations.neo4j import VertectorTextSplitter

        splitter = VertectorTextSplitter(loader=mock_loader)
        splitter.last_chunks = mock_chunks

        # Index too high
        assert splitter.get_chunk_metadata(10) == {}

        # Negative index
        assert splitter.get_chunk_metadata(-1) == {}

    async def test_stores_last_chunks(self, mock_loader, mock_chunking_result):
        """Test run() stores chunks in last_chunks attribute."""
        from vertector_data_ingestion.integrations.neo4j import VertectorTextSplitter

        splitter = VertectorTextSplitter(loader=mock_loader)
        splitter.chunker.chunk_document = MagicMock(return_value=mock_chunking_result)

        await splitter.run("text")

        assert len(splitter.last_chunks) == 2
        assert splitter.last_chunks == mock_chunking_result.chunks
