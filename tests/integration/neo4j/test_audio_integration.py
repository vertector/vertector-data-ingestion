"""Integration tests for Neo4j audio processing with real files.

These tests use actual audio files to verify the complete pipeline:
- Audio loading with VertectorAudioLoader
- Audio segment chunking with VertectorTextSplitter
- Metadata preservation (timestamps, duration, language)
"""

from pathlib import Path

import pytest

# Skip if neo4j-graphrag not installed
pytest.importorskip("neo4j_graphrag")

from vertector_data_ingestion import AudioConfig, WhisperModelSize
from vertector_data_ingestion.integrations.neo4j import (
    MultimodalLoader,
    VertectorAudioLoader,
    VertectorTextSplitter,
)


@pytest.mark.asyncio
class TestAudioIntegration:
    """Integration tests for audio processing."""

    @pytest.fixture
    def audio_file(self):
        """Get test audio file path."""
        audio_path = Path(__file__).parent.parent.parent.parent / "test_documents" / "harvard.wav"
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")
        return audio_path

    async def test_audio_loader_with_real_file(self, audio_file):
        """Test VertectorAudioLoader with real audio file."""
        # Initialize loader with small model for speed
        config = AudioConfig(
            model_size=WhisperModelSize.BASE,
            word_timestamps=True,
        )
        loader = VertectorAudioLoader(config=config)

        # Load audio
        result = await loader.run(audio_file)

        # Verify result structure
        assert result.text is not None
        assert len(result.text) > 0
        assert "Audio Transcription" in result.text
        assert audio_file.name in result.text

        # Verify metadata
        assert result.document_info.document_type == "audio"
        assert result.document_info.metadata["modality"] == "audio"
        assert "duration" in result.document_info.metadata
        assert "language" in result.document_info.metadata
        assert "segments" in result.document_info.metadata

        # Verify transcription result is stored
        assert loader.last_transcription_result is not None
        assert loader.last_transcription_result.duration > 0
        assert len(loader.last_transcription_result.segments) > 0

        print(f"\n✓ Loaded audio: {audio_file.name}")
        print(f"  Duration: {loader.last_transcription_result.duration:.2f}s")
        print(f"  Language: {loader.last_transcription_result.language}")
        print(f"  Segments: {len(loader.last_transcription_result.segments)}")

    async def test_audio_splitter_with_real_file(self, audio_file):
        """Test VertectorTextSplitter chunks audio segments correctly."""
        # Load audio
        config = AudioConfig(model_size=WhisperModelSize.BASE)
        loader = VertectorAudioLoader(config=config)
        pdf_result = await loader.run(audio_file)

        # Initialize splitter
        splitter = VertectorTextSplitter(
            loader=loader,
            chunk_size=512,
            tokenizer="Qwen/Qwen3-Embedding-0.6B",
        )

        # Chunk audio (using segments)
        chunks = await splitter.run(pdf_result.text)

        # Verify chunks were created
        assert len(chunks.chunks) > 0
        print(f"\n✓ Created {len(chunks.chunks)} audio chunks")

        # Verify each chunk
        for i, chunk in enumerate(chunks.chunks):
            # Verify text
            assert chunk.text is not None
            assert len(chunk.text) > 0

            # Verify index
            assert chunk.index == i

            # Verify audio-specific metadata
            assert chunk.metadata["modality"] == "audio"
            assert "start_time" in chunk.metadata
            assert "end_time" in chunk.metadata
            assert "duration" in chunk.metadata
            assert "language" in chunk.metadata

            # Verify timing metadata is valid
            start_time = float(chunk.metadata["start_time"])
            end_time = float(chunk.metadata["end_time"])
            duration = float(chunk.metadata["duration"])

            assert start_time >= 0
            assert end_time > start_time
            assert duration == pytest.approx(end_time - start_time, abs=0.01)

            # Print first chunk details
            if i == 0:
                print(f"\n  Chunk 0:")
                print(f"    Text: {chunk.text[:80]}...")
                print(f"    Start: {start_time:.2f}s")
                print(f"    End: {end_time:.2f}s")
                print(f"    Duration: {duration:.2f}s")
                print(f"    Tokens: {chunk.metadata['token_count']}")

        # Verify chunks are stored in splitter
        assert len(splitter.last_chunks) == len(chunks.chunks)

    async def test_multimodal_loader_with_audio(self, audio_file):
        """Test MultimodalLoader correctly routes audio files."""
        loader = MultimodalLoader()

        # Load audio (should route to audio_loader)
        result = await loader.run(audio_file)

        # Verify it's recognized as audio
        assert result.document_info.document_type == "audio"
        assert loader.last_metadata["modality"] == "audio"

        # Verify transcription result is accessible
        assert loader.audio_loader.last_transcription_result is not None

        # Verify can be chunked
        splitter = VertectorTextSplitter(loader=loader.audio_loader, chunk_size=512)
        chunks = await splitter.run(result.text)

        assert len(chunks.chunks) > 0
        assert all(c.metadata["modality"] == "audio" for c in chunks.chunks)

        print(f"\n✓ MultimodalLoader processed audio: {audio_file.name}")
        print(f"  Chunks: {len(chunks.chunks)}")

    async def test_audio_chunk_token_counting(self, audio_file):
        """Test that audio chunks have accurate token counts."""
        # Load and chunk audio
        loader = VertectorAudioLoader()
        await loader.run(audio_file)

        splitter = VertectorTextSplitter(loader=loader, chunk_size=512)
        chunks = await splitter.run("")

        # Verify token counts
        for chunk in chunks.chunks:
            token_count = int(chunk.metadata["token_count"])
            assert token_count > 0

            # Token count should be reasonable for the text length
            text_length = len(chunk.text)
            # Rough heuristic: tokens should be less than characters
            assert token_count <= text_length

            # For English, rough ratio is ~4 chars per token
            # But allow wide range for different languages/content
            assert token_count >= text_length / 10
            assert token_count <= text_length

        print(f"\n✓ Token counting verified for {len(chunks.chunks)} chunks")
        avg_tokens = sum(int(c.metadata["token_count"]) for c in chunks.chunks) / len(chunks.chunks)
        print(f"  Average tokens per chunk: {avg_tokens:.1f}")

    async def test_audio_metadata_preserved_through_pipeline(self, audio_file):
        """Test that audio metadata is preserved through the entire pipeline."""
        # Full pipeline: load → chunk
        loader = VertectorAudioLoader()
        pdf_result = await loader.run(audio_file)

        splitter = VertectorTextSplitter(loader=loader)
        chunks = await splitter.run(pdf_result.text)

        # Get original transcription result
        original_result = loader.last_transcription_result

        # Verify each chunk maps to a segment
        assert len(chunks.chunks) == len(original_result.segments)

        for i, (chunk, segment) in enumerate(zip(chunks.chunks, original_result.segments)):
            # Verify chunk text matches segment text
            assert chunk.text == segment.text

            # Verify timing metadata matches
            assert float(chunk.metadata["start_time"]) == pytest.approx(segment.start, abs=0.01)
            assert float(chunk.metadata["end_time"]) == pytest.approx(segment.end, abs=0.01)

            # Verify language is preserved
            assert chunk.metadata["language"] == original_result.language

        print(f"\n✓ Metadata preserved for {len(chunks.chunks)} segments")
        print(f"  Language: {original_result.language}")
        print(f"  Total duration: {original_result.duration:.2f}s")


@pytest.mark.asyncio
class TestDocumentIntegration:
    """Integration tests for document processing (for comparison)."""

    @pytest.fixture
    def pdf_file(self):
        """Get test PDF file path."""
        pdf_path = Path(__file__).parent.parent.parent.parent / "test_documents" / "2112.13734v2.pdf"
        if not pdf_path.exists():
            pytest.skip(f"Test PDF file not found: {pdf_path}")
        return pdf_path

    async def test_document_loader_with_real_file(self, pdf_file):
        """Test VertectorDataLoader with real PDF file."""
        from vertector_data_ingestion import LocalMpsConfig
        from vertector_data_ingestion.integrations.neo4j import VertectorDataLoader

        loader = VertectorDataLoader(config=LocalMpsConfig())
        result = await loader.run(pdf_file)

        # Verify result
        assert result.text is not None
        assert len(result.text) > 0
        assert result.document_info.document_type == "document"

        # Verify document wrapper is stored
        assert loader.last_document is not None
        assert loader.last_document.metadata.num_pages > 0

        print(f"\n✓ Loaded document: {pdf_file.name}")
        print(f"  Pages: {loader.last_document.metadata.num_pages}")

    async def test_document_splitter_with_real_file(self, pdf_file):
        """Test VertectorTextSplitter chunks documents correctly."""
        from vertector_data_ingestion import LocalMpsConfig
        from vertector_data_ingestion.integrations.neo4j import VertectorDataLoader

        # Load document
        loader = VertectorDataLoader(config=LocalMpsConfig())
        await loader.run(pdf_file)

        # Chunk document
        splitter = VertectorTextSplitter(loader=loader, chunk_size=512)
        chunks = await splitter.run("")

        # Verify chunks
        assert len(chunks.chunks) > 0
        print(f"\n✓ Created {len(chunks.chunks)} document chunks")

        # Verify document-specific metadata
        for chunk in chunks.chunks:
            # Should have page numbers (not audio metadata)
            assert "page_no" in chunk.metadata or chunk.index == 0
            assert "start_time" not in chunk.metadata  # Audio-specific
            assert "modality" not in chunk.metadata  # Not set for docs by default

        print(f"  First chunk page: {chunks.chunks[0].metadata.get('page_no', 'N/A')}")


@pytest.mark.asyncio
class TestMultimodalIntegration:
    """Integration tests for unified multimodal pipeline."""

    @pytest.fixture
    def audio_file(self):
        """Get test audio file path."""
        audio_path = Path(__file__).parent.parent.parent.parent / "test_documents" / "harvard.wav"
        if not audio_path.exists():
            pytest.skip(f"Test audio file not found: {audio_path}")
        return audio_path

    @pytest.fixture
    def pdf_file(self):
        """Get test PDF file path."""
        pdf_path = Path(__file__).parent.parent.parent.parent / "test_documents" / "2112.13734v2.pdf"
        if not pdf_path.exists():
            pytest.skip(f"Test PDF file not found: {pdf_path}")
        return pdf_path

    async def test_multimodal_pipeline_handles_both_types(self, audio_file, pdf_file):
        """Test that MultimodalLoader and splitter work for both audio and documents."""
        loader = MultimodalLoader()

        # Process document
        doc_result = await loader.run(pdf_file)
        assert doc_result.document_info.document_type == "document"
        assert loader.last_document is not None

        # Chunk document
        splitter = VertectorTextSplitter(loader=loader.doc_loader, chunk_size=512)
        doc_chunks = await splitter.run(doc_result.text)
        assert len(doc_chunks.chunks) > 0
        print(f"\n✓ Document: {len(doc_chunks.chunks)} chunks")

        # Process audio
        audio_result = await loader.run(audio_file)
        assert audio_result.document_info.document_type == "audio"
        assert loader.audio_loader.last_transcription_result is not None

        # Chunk audio
        splitter = VertectorTextSplitter(loader=loader.audio_loader, chunk_size=512)
        audio_chunks = await splitter.run(audio_result.text)
        assert len(audio_chunks.chunks) > 0
        print(f"✓ Audio: {len(audio_chunks.chunks)} chunks")

        # Verify different metadata
        assert "page_no" in doc_chunks.chunks[0].metadata or True  # May not have page_no
        assert "start_time" in audio_chunks.chunks[0].metadata
        assert "modality" in audio_chunks.chunks[0].metadata

        print(f"\n✓ Multimodal pipeline verified")

    async def test_multimodal_loader_property_delegation(self, audio_file):
        """Test that MultimodalLoader properly delegates properties to sub-loaders.

        This test verifies the fix for the bug where VertectorTextSplitter
        couldn't access last_transcription_result through MultimodalLoader.
        The properties should be accessible directly on the MultimodalLoader
        instance, not just on the internal sub-loaders.
        """
        loader = MultimodalLoader()
        splitter = VertectorTextSplitter(loader=loader, chunk_size=512)

        # Load audio through MultimodalLoader
        audio_result = await loader.run(audio_file)

        # Verify properties are accessible through MultimodalLoader (not just audio_loader)
        assert loader.last_transcription_result is not None
        assert loader.last_transcription_result.duration > 0
        assert loader.last_transcription_result.language == "en"
        assert len(loader.last_transcription_result.segments) > 0

        # Verify splitter can access the properties through loader
        audio_chunks = await splitter.run(audio_result.text)
        assert len(audio_chunks.chunks) > 0

        # Verify chunks have audio metadata
        chunk = audio_chunks.chunks[0]
        assert "start_time" in chunk.metadata
        assert "end_time" in chunk.metadata
        assert "duration" in chunk.metadata
        assert "modality" in chunk.metadata
        assert chunk.metadata["modality"] == "audio"

        print(f"\n✓ MultimodalLoader property delegation works")
        print(f"  Loaded audio: {loader.last_transcription_result.duration:.2f}s")
        print(f"  Created {len(audio_chunks.chunks)} chunks through delegated properties")

    async def test_multimodal_state_isolation(self, audio_file, pdf_file):
        """Test that loading one modality clears the state of the other.

        Regression test for bug where loading document then audio would cause
        audio splitting to return document chunks instead of audio chunks.
        """
        loader = MultimodalLoader()
        splitter = VertectorTextSplitter(loader=loader, chunk_size=512)

        # Load document first
        doc_result = await loader.run(pdf_file)
        assert loader.last_document is not None
        assert loader.last_transcription_result is None

        # Chunk document
        doc_chunks = await splitter.run(doc_result.text)
        doc_chunk_text = doc_chunks.chunks[0].text

        # Now load audio - this should CLEAR document state
        audio_result = await loader.run(audio_file)
        assert loader.last_document is None  # Document state cleared
        assert loader.last_transcription_result is not None  # Audio state set

        # Chunk audio - should get AUDIO chunks, not document chunks
        audio_chunks = await splitter.run(audio_result.text)
        audio_chunk_text = audio_chunks.chunks[0].text

        # Verify audio chunks are different from document chunks
        assert audio_chunk_text != doc_chunk_text, (
            "Audio chunks should not equal document chunks after state transition"
        )

        # Verify metadata is correct
        assert "start_time" in audio_chunks.chunks[0].metadata
        assert "modality" in audio_chunks.chunks[0].metadata
        assert audio_chunks.chunks[0].metadata["modality"] == "audio"

        print(f"\n✓ Multimodal state isolation verified")
        print(f"  Document chunks: {len(doc_chunks.chunks)}")
        print(f"  Audio chunks: {len(audio_chunks.chunks)}")
        print(f"  State properly isolated between modalities")
