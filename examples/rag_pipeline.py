"""Complete RAG pipeline example."""

from pathlib import Path

from vertector_data_ingestion import (
    UniversalConverter,
    LocalMpsConfig,
    HybridChunker,
    ChromaAdapter,
    ExportFormat,
    setup_logging,
)


def main():
    """Demonstrate complete RAG pipeline."""

    setup_logging(log_level="INFO")

    # Example 1: Basic RAG Pipeline
    print("=" * 60)
    print("Example 1: Basic RAG Pipeline")
    print("=" * 60)

    # Initialize converter
    config = LocalMpsConfig()
    converter = UniversalConverter(config)

    # Convert document
    doc_path = Path("path/to/your/document.pdf")

    if doc_path.exists():
        doc = converter.convert_single(doc_path)

        print(f"\nDocument: {doc.metadata.source_path.name}")
        print(f"Pages: {doc.metadata.num_pages}")
        print(f"Pipeline: {doc.metadata.pipeline_used}")

        # Create chunks
        chunker = HybridChunker()
        chunks = chunker.chunk_document(doc.document)

        print(f"\nChunking Results:")
        print(f"  Total chunks: {chunks.total_chunks}")
        print(f"  Total tokens: {chunks.total_tokens}")
        print(f"  Avg chunk size: {chunks.avg_chunk_size:.1f} tokens")

        # Store in vector database
        vector_store = ChromaAdapter(
            collection_name="my_documents",
            persist_directory=Path("./chroma_db")
        )

        vector_store.add_chunks(chunks.chunks)
        print(f"\nStored {len(chunks.chunks)} chunks in ChromaDB")

        # Search
        query = "What is the main topic?"
        results = vector_store.search(query, top_k=3)

        print(f"\nSearch Results for: '{query}'")
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Score: {result['score']:.3f}")
            print(f"  Page: {result['metadata'].get('page_no', 'N/A')}")
            print(f"  Text: {result['text'][:150]}...")

    else:
        print(f"Document not found: {doc_path}")
        demonstrate_without_file()


def demonstrate_without_file():
    """Demonstrate RAG concepts without actual file."""

    print("\n" + "=" * 60)
    print("RAG Pipeline Components")
    print("=" * 60)

    # Show chunking strategies
    from vertector_data_ingestion.models.config import ChunkingConfig

    print("\n1. Chunking Strategies:")

    # Small chunks for precise retrieval
    precise_config = ChunkingConfig(
        max_tokens=256,
        overlap_tokens=25,
        respect_boundaries=True,
    )
    print(f"   Precise retrieval: {precise_config.max_tokens} tokens")

    # Large chunks for more context
    context_config = ChunkingConfig(
        max_tokens=1024,
        overlap_tokens=100,
        respect_boundaries=True,
    )
    print(f"   More context: {context_config.max_tokens} tokens")

    # Balanced approach
    balanced_config = ChunkingConfig(
        max_tokens=512,
        overlap_tokens=50,
        respect_boundaries=True,
    )
    print(f"   Balanced: {balanced_config.max_tokens} tokens")


def batch_processing_example():
    """Example of processing multiple documents for RAG."""

    print("\n" + "=" * 60)
    print("Example 2: Batch Processing for RAG")
    print("=" * 60)

    converter = UniversalConverter()
    chunker = HybridChunker()

    # Process multiple documents
    documents = [
        Path("doc1.pdf"),
        Path("doc2.docx"),
        Path("doc3.pptx"),
    ]

    all_chunks = []

    for doc_path in documents:
        if doc_path.exists():
            print(f"\nProcessing: {doc_path.name}")

            # Convert
            doc = converter.convert_single(doc_path)

            # Chunk
            chunks = chunker.chunk_document(doc.document)

            # Add source metadata
            for chunk in chunks.chunks:
                chunk.metadata["source_file"] = doc_path.name

            all_chunks.extend(chunks.chunks)

            print(f"  Created {chunks.total_chunks} chunks")

    # Store all chunks
    if all_chunks:
        vector_store = ChromaAdapter(collection_name="document_collection")
        vector_store.add_chunks(all_chunks)

        print(f"\nTotal chunks stored: {len(all_chunks)}")


def advanced_search_example():
    """Example of advanced vector search."""

    print("\n" + "=" * 60)
    print("Example 3: Advanced Vector Search")
    print("=" * 60)

    vector_store = ChromaAdapter(collection_name="my_documents")

    # Search with different strategies
    queries = [
        "What are the key findings?",
        "Explain the methodology",
        "What are the conclusions?",
    ]

    for query in queries:
        print(f"\nQuery: '{query}'")

        # Standard search
        results = vector_store.search(query, top_k=3)

        for i, result in enumerate(results, 1):
            print(f"  {i}. Score: {result['score']:.3f}")
            print(f"     Text: {result['text'][:100]}...")

        # Search with metadata filter (if applicable)
        # filtered_results = vector_store.search(
        #     query,
        #     top_k=3,
        #     filter={"source_file": "doc1.pdf"}
        # )


def export_chunks_example():
    """Example of exporting chunks for inspection."""

    print("\n" + "=" * 60)
    print("Example 4: Export Chunks")
    print("=" * 60)

    converter = UniversalConverter()
    chunker = HybridChunker()

    doc_path = Path("path/to/document.pdf")

    if doc_path.exists():
        doc = converter.convert_single(doc_path)
        chunks = chunker.chunk_document(doc.document)

        # Export chunks to JSON for inspection
        import json

        output = {
            "document": doc_path.name,
            "total_chunks": chunks.total_chunks,
            "total_tokens": chunks.total_tokens,
            "chunks": [
                {
                    "id": chunk.chunk_id,
                    "text": chunk.text,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks.chunks
            ]
        }

        output_file = Path("chunks_export.json")
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        print(f"Exported chunks to {output_file}")


def retrieval_qa_example():
    """Example of retrieval-based QA."""

    print("\n" + "=" * 60)
    print("Example 5: Retrieval-Based QA")
    print("=" * 60)

    vector_store = ChromaAdapter(collection_name="my_documents")

    # Simulate Q&A workflow
    questions = [
        "What is the main argument?",
        "What evidence supports this?",
        "What are the limitations?",
    ]

    for question in questions:
        print(f"\nQuestion: {question}")

        # Retrieve relevant context
        results = vector_store.search(question, top_k=3)

        # Combine top results for context
        context = "\n\n".join([
            f"[Page {r['metadata'].get('page_no', '?')}] {r['text']}"
            for r in results
        ])

        print("\nRetrieved Context:")
        print(context[:300] + "...")

        # In production, you would pass this context to an LLM
        # answer = llm.generate(question=question, context=context)


def cleanup_example():
    """Example of cleaning up vector store."""

    print("\n" + "=" * 60)
    print("Example 6: Vector Store Management")
    print("=" * 60)

    # List collections
    print("Managing vector stores...")

    # Create new collection
    new_store = ChromaAdapter(collection_name="temp_collection")
    print(f"Created collection: temp_collection")

    # Delete when done
    new_store.delete_collection()
    print(f"Deleted collection: temp_collection")


if __name__ == "__main__":
    main()
    batch_processing_example()
    advanced_search_example()
    export_chunks_example()
    retrieval_qa_example()
    cleanup_example()
