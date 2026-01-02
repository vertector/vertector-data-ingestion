"""
Examples of using different embedding models and tokenizers.

This demonstrates how to configure the system with various models:
- Qwen models (auto-detected left padding)
- SentenceTransformers models
- Other Hugging Face models
"""

from pathlib import Path

from vertector_data_ingestion import (
    UniversalConverter,
    HybridChunker,
    ChromaAdapter,
    LocalMpsConfig,
)
from vertector_data_ingestion.models.config import ChunkingConfig, VectorStoreConfig


# Example 1: Using default Qwen models (high quality, large context)
def example_qwen_default():
    """Default configuration with Qwen3-Embedding-0.6B."""
    print("\n=== Example 1: Default Qwen3 Model ===")

    # Default config uses Qwen/Qwen3-Embedding-0.6B for both chunking and embeddings
    chunker = HybridChunker()  # Auto-detects left padding for Qwen
    vector_store = ChromaAdapter(collection_name="qwen_default")

    print(f"Tokenizer: {chunker.config.tokenizer}")
    print(f"Embedding model: Qwen/Qwen3-Embedding-0.6B")
    print(f"Padding side: left (auto-detected)")


# Example 2: Using SentenceTransformers models (fast, small)
def example_sentence_transformers():
    """Using a popular SentenceTransformers model."""
    print("\n=== Example 2: SentenceTransformers (all-MiniLM-L6-v2) ===")

    # Configure with SentenceTransformers model
    chunk_config = ChunkingConfig(
        tokenizer="sentence-transformers/all-MiniLM-L6-v2",
        max_tokens=512,  # This model has 512 max context
    )

    vector_config = VectorStoreConfig(
        embedding_model="sentence-transformers/all-MiniLM-L6-v2"
    )

    chunker = HybridChunker(config=chunk_config)  # Auto-detects right padding
    vector_store = ChromaAdapter(
        collection_name="minilm",
        embedding_model=vector_config.embedding_model,
    )

    print(f"Tokenizer: {chunk_config.tokenizer}")
    print(f"Embedding model: {vector_config.embedding_model}")
    print(f"Padding side: right (auto-detected)")
    print(f"Max tokens: {chunk_config.max_tokens}")


# Example 3: Using E5 models (balanced quality/speed)
def example_e5_large():
    """Using the E5 large model for better quality."""
    print("\n=== Example 3: E5 Large (intfloat/e5-large-v2) ===")

    chunk_config = ChunkingConfig(
        tokenizer="intfloat/e5-large-v2",
        max_tokens=512,
    )

    vector_config = VectorStoreConfig(
        embedding_model="intfloat/e5-large-v2"
    )

    chunker = HybridChunker(config=chunk_config)
    vector_store = ChromaAdapter(
        collection_name="e5_large",
        embedding_model=vector_config.embedding_model,
    )

    print(f"Tokenizer: {chunk_config.tokenizer}")
    print(f"Embedding model: {vector_config.embedding_model}")
    print(f"Padding side: right (auto-detected)")


# Example 4: Using BGE models (high quality English)
def example_bge_large():
    """Using BGE large model for high quality English embeddings."""
    print("\n=== Example 4: BGE Large (BAAI/bge-large-en-v1.5) ===")

    chunk_config = ChunkingConfig(
        tokenizer="BAAI/bge-large-en-v1.5",
        max_tokens=512,
    )

    vector_config = VectorStoreConfig(
        embedding_model="BAAI/bge-large-en-v1.5"
    )

    chunker = HybridChunker(config=chunk_config)
    vector_store = ChromaAdapter(
        collection_name="bge_large",
        embedding_model=vector_config.embedding_model,
    )

    print(f"Tokenizer: {chunk_config.tokenizer}")
    print(f"Embedding model: {vector_config.embedding_model}")
    print(f"Padding side: right (auto-detected)")


# Example 5: Multilingual with BGE-M3
def example_multilingual():
    """Using BGE-M3 for multilingual embeddings."""
    print("\n=== Example 5: Multilingual (BAAI/bge-m3) ===")

    chunk_config = ChunkingConfig(
        tokenizer="BAAI/bge-m3",
        max_tokens=8192,  # BGE-M3 supports 8192 context
    )

    vector_config = VectorStoreConfig(
        embedding_model="BAAI/bge-m3"
    )

    chunker = HybridChunker(config=chunk_config)
    vector_store = ChromaAdapter(
        collection_name="bge_m3",
        embedding_model=vector_config.embedding_model,
    )

    print(f"Tokenizer: {chunk_config.tokenizer}")
    print(f"Embedding model: {vector_config.embedding_model}")
    print(f"Padding side: right (auto-detected)")
    print(f"Max tokens: {chunk_config.max_tokens}")
    print("Supports 100+ languages!")


# Example 6: Using environment variables
def example_env_config():
    """Configure models via environment variables."""
    print("\n=== Example 6: Environment Variable Configuration ===")
    print("\nSet these environment variables:")
    print("  export VERTECTOR_CHUNK_TOKENIZER='sentence-transformers/all-MiniLM-L6-v2'")
    print("  export VERTECTOR_CHUNK_MAX_TOKENS=512")
    print("  export VERTECTOR_EMBEDDING_MODEL='sentence-transformers/all-MiniLM-L6-v2'")
    print("\nThen use default configs (they'll read from environment):")
    print("  chunker = HybridChunker()")
    print("  vector_store = ChromaAdapter(collection_name='my_collection')")


# Example 7: Complete RAG pipeline with custom model
def example_complete_pipeline():
    """Complete RAG pipeline with a different model."""
    print("\n=== Example 7: Complete Pipeline with Custom Model ===")

    # Use E5 large for better quality
    chunk_config = ChunkingConfig(
        tokenizer="intfloat/e5-large-v2",
        max_tokens=512,
    )

    # Create components
    converter = UniversalConverter(LocalMpsConfig())
    chunker = HybridChunker(config=chunk_config)
    vector_store = ChromaAdapter(
        collection_name="e5_pipeline",
        embedding_model="intfloat/e5-large-v2",
    )

    # Process document
    doc_path = Path("../test_documents/arxiv_sample.pdf")
    if doc_path.exists():
        # Convert
        doc = converter.convert(doc_path)
        print(f"Converted: {doc.metadata.source_path.name}")

        # Chunk
        result = chunker.chunk_document(doc)
        print(f"Created {result.total_chunks} chunks")

        # Store
        vector_store.add_chunks(result.chunks)
        print(f"Stored chunks in vector database")

        # Search
        search_results = vector_store.search("machine learning", top_k=3)
        print(f"\nFound {len(search_results)} relevant chunks")
    else:
        print(f"Document not found: {doc_path}")


if __name__ == "__main__":
    # Run all examples
    example_qwen_default()
    example_sentence_transformers()
    example_e5_large()
    example_bge_large()
    example_multilingual()
    example_env_config()
    example_complete_pipeline()

    print("\n" + "=" * 60)
    print("Model Selection Guide:")
    print("=" * 60)
    print("\n1. **Qwen/Qwen3-Embedding-0.6B** (default)")
    print("   - High quality embeddings")
    print("   - 32,768 token context window")
    print("   - Requires left padding (auto-detected)")
    print("   - Larger model, slower but more accurate")
    print("\n2. **sentence-transformers/all-MiniLM-L6-v2**")
    print("   - Fast and lightweight")
    print("   - 512 token context")
    print("   - Great for general use")
    print("   - Best for speed")
    print("\n3. **intfloat/e5-large-v2**")
    print("   - Balanced quality/speed")
    print("   - 512 token context")
    print("   - Good general purpose")
    print("\n4. **BAAI/bge-large-en-v1.5**")
    print("   - High quality English")
    print("   - 512 token context")
    print("   - Best for English-only documents")
    print("\n5. **BAAI/bge-m3**")
    print("   - Multilingual (100+ languages)")
    print("   - 8,192 token context")
    print("   - Best for non-English or mixed language")
