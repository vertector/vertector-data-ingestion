"""Basic usage example for Vertector Data Ingestion."""

from pathlib import Path

from vertector_data_ingestion import (
    UniversalConverter,
    ConverterConfig,
    HybridChunker,
    ChromaAdapter,
    ExportFormat,
    setup_logging,
)


def main():
    """Demonstrate basic usage of the pipeline."""
    
    # Setup logging
    setup_logging(log_level="INFO")
    
    # Initialize converter with default config
    print("Initializing converter...")
    config = ConverterConfig()
    converter = UniversalConverter(config)
    
    # Convert a document
    print("\nConverting document...")
    source = Path("path/to/your/document.pdf")
    
    if source.exists():
        # Convert document
        doc_wrapper = converter.convert_single(source)
        
        print(f"Converted {doc_wrapper.metadata.num_pages} pages")
        print(f"Processing time: {doc_wrapper.metadata.processing_time:.2f}s")
        
        # Export to different formats
        print("\nExporting to formats...")
        
        # Markdown
        markdown = converter.export(doc_wrapper, ExportFormat.MARKDOWN)
        print(f"Markdown length: {len(markdown)} characters")
        
        # JSON
        json_output = converter.export(doc_wrapper, ExportFormat.JSON)
        print(f"JSON length: {len(json_output)} characters")
        
        # Save to file
        output_path = Path("output/document.md")
        converter.convert_and_export(
            source=source,
            output_path=output_path,
            format=ExportFormat.MARKDOWN,
        )
        print(f"Saved to: {output_path}")
        
        # Chunk for RAG
        print("\nChunking for RAG...")
        chunker = HybridChunker(config.chunking)
        chunking_result = chunker.chunk_document(doc_wrapper)
        
        print(f"Created {chunking_result.total_chunks} chunks")
        print(f"Total tokens: {chunking_result.total_tokens}")
        print(f"Avg chunk size: {chunking_result.avg_chunk_size:.1f} tokens")
        
        # Store in vector database
        print("\nStoring in ChromaDB...")
        chroma = ChromaAdapter(
            collection_name="my_documents",
            persist_directory=Path("./chroma_db"),
        )
        
        chroma.add_chunks(chunking_result.chunks)
        
        # Search
        print("\nSearching...")
        results = chroma.search("your search query", top_k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\nResult {i}:")
            print(f"  Text: {result['text'][:100]}...")
            print(f"  Page: {result['metadata'].get('page_no', 'N/A')}")
            
    else:
        print(f"Document not found: {source}")
        print("\nExample: How to use the pipeline without a file")
        
        # Show hardware detection
        from vertector_data_ingestion import HardwareDetector
        
        hw_info = HardwareDetector.get_device_info()
        print("\nHardware Detection:")
        for key, value in hw_info.items():
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
