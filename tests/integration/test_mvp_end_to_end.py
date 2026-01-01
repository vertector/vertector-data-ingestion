"""
End-to-End MVP Test for Vertector Data Ingestion Pipeline

Tests the complete workflow:
1. Document ingestion (PDF, DOCX, PPTX, XLSX)
2. Export to all formats (Markdown, JSON, DocTags)
3. Chunking for RAG
4. Save all outputs to files
"""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vertector_data_ingestion.core.universal_converter import UniversalConverter
from vertector_data_ingestion.models.config import ConverterConfig, ExportFormat
from vertector_data_ingestion.chunkers.hybrid_chunker import HybridChunker

# Create output directory
OUTPUT_DIR = Path("mvp_test_output")
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("VERTECTOR DATA INGESTION - END-TO-END MVP TEST")
print("=" * 80)
print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output Directory: {OUTPUT_DIR}")
print()

# Test documents
test_docs = [
    {"name": "PDF (Research Paper)", "path": Path("test_documents/arxiv_sample.pdf")},
    {"name": "DOCX (Word Document)", "path": Path("test_documents/sample_docx.docx")},
    {"name": "PPTX (PowerPoint)", "path": Path("test_documents/sample_pptx.pptx")},
    {"name": "XLSX (Excel Spreadsheet)", "path": Path("test_documents/sample_xlsx.xlsx")},
]

# Initialize converter
print("1. Initializing Pipeline...")
print("-" * 80)
config = ConverterConfig()
converter = UniversalConverter(config)
chunker = HybridChunker()
print(f"✅ Pipeline initialized on {converter.hardware_config.device_type.value}")
print(f"   Batch size: {converter.hardware_config.batch_size}")
print(f"   Workers: {converter.hardware_config.num_workers}")
print()

# Process each document
results = []
for idx, doc_info in enumerate(test_docs, 1):
    print(f"{idx}. Processing {doc_info['name']}")
    print("-" * 80)

    if not doc_info["path"].exists():
        print(f"   ❌ File not found: {doc_info['path']}")
        results.append({"doc": doc_info["name"], "status": "SKIPPED", "reason": "File not found"})
        print()
        continue

    try:
        # Step 1: Ingest document
        print(f"   Step 1/4: Ingesting document...")
        doc_wrapper = converter.convert_single(doc_info["path"])
        print(f"   ✅ Ingested: {doc_wrapper.metadata.num_pages} pages in {doc_wrapper.metadata.processing_time:.2f}s")
        print(f"      Pipeline: {doc_wrapper.metadata.pipeline_type}")

        # Step 2: Export to all formats
        print(f"   Step 2/4: Exporting to all formats...")

        # Create document-specific output directory
        doc_output_dir = OUTPUT_DIR / doc_info["path"].stem
        doc_output_dir.mkdir(exist_ok=True)

        # Export Markdown
        markdown_content = converter.export(doc_wrapper, ExportFormat.MARKDOWN)
        markdown_file = doc_output_dir / f"{doc_info['path'].stem}.md"
        markdown_file.write_text(markdown_content, encoding="utf-8")
        print(f"   ✅ Markdown: {len(markdown_content):,} chars → {markdown_file}")

        # Export JSON
        json_content = converter.export(doc_wrapper, ExportFormat.JSON)
        json_file = doc_output_dir / f"{doc_info['path'].stem}.json"
        json_file.write_text(json_content, encoding="utf-8")
        print(f"   ✅ JSON: {len(json_content):,} chars → {json_file}")

        # Export DocTags
        doctags_content = converter.export(doc_wrapper, ExportFormat.DOCTAGS)
        doctags_file = doc_output_dir / f"{doc_info['path'].stem}.doctags"
        doctags_file.write_text(doctags_content, encoding="utf-8")
        print(f"   ✅ DocTags: {len(doctags_content):,} chars → {doctags_file}")

        # Step 3: Chunk for RAG
        print(f"   Step 3/4: Chunking for RAG...")
        chunking_result = chunker.chunk_document(doc_wrapper)
        num_chunks = len(chunking_result.chunks)

        # Calculate stats
        if num_chunks > 0:
            token_counts = [chunk.token_count for chunk in chunking_result.chunks]
            avg_tokens = sum(token_counts) / len(token_counts)
            min_tokens = min(token_counts)
            max_tokens = max(token_counts)

            print(f"   ✅ Created {num_chunks} chunks")
            print(f"      Avg tokens/chunk: {avg_tokens:.1f}")
            print(f"      Token range: {min_tokens}-{max_tokens}")

            # Step 4: Save chunks to file
            print(f"   Step 4/4: Saving chunks...")
            chunks_file = doc_output_dir / f"{doc_info['path'].stem}_chunks.txt"
            with chunks_file.open("w", encoding="utf-8") as f:
                f.write(f"Document: {doc_info['name']}\n")
                f.write(f"Total Chunks: {num_chunks}\n")
                f.write(f"Avg Tokens: {avg_tokens:.1f}\n")
                f.write("=" * 80 + "\n\n")

                for i, chunk in enumerate(chunking_result.chunks, 1):
                    f.write(f"CHUNK {i}/{num_chunks}\n")
                    f.write(f"ID: {chunk.chunk_id}\n")
                    f.write(f"Tokens: {chunk.token_count}\n")
                    if chunk.metadata.get("page_no"):
                        f.write(f"Page: {chunk.metadata['page_no']}\n")
                    f.write("-" * 80 + "\n")
                    f.write(chunk.text + "\n")
                    f.write("=" * 80 + "\n\n")

            print(f"   ✅ Chunks saved → {chunks_file}")
        else:
            print(f"   ⚠️  No chunks created")

        results.append({
            "doc": doc_info["name"],
            "status": "SUCCESS",
            "pages": doc_wrapper.metadata.num_pages,
            "time": doc_wrapper.metadata.processing_time,
            "chunks": num_chunks,
            "output_dir": str(doc_output_dir)
        })

    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        results.append({"doc": doc_info["name"], "status": "FAILED", "error": str(e)})

    print()

# Summary
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print()

successful = [r for r in results if r["status"] == "SUCCESS"]
failed = [r for r in results if r["status"] == "FAILED"]
skipped = [r for r in results if r["status"] == "SKIPPED"]

print(f"📊 Results: {len(successful)}/{len(test_docs)} successful")
print()

if successful:
    print("✅ Successful Documents:")
    for result in successful:
        print(f"   • {result['doc']}")
        print(f"     Pages: {result['pages']}, Time: {result['time']:.2f}s, Chunks: {result['chunks']}")
        print(f"     Output: {result['output_dir']}")
    print()

if failed:
    print("❌ Failed Documents:")
    for result in failed:
        print(f"   • {result['doc']}: {result.get('error', 'Unknown error')}")
    print()

if skipped:
    print("⚠️  Skipped Documents:")
    for result in skipped:
        print(f"   • {result['doc']}: {result.get('reason', 'Unknown reason')}")
    print()

# List all generated files
print("=" * 80)
print("GENERATED FILES")
print("=" * 80)
print()

all_files = sorted(OUTPUT_DIR.rglob("*"))
file_count = sum(1 for f in all_files if f.is_file())

print(f"Total files generated: {file_count}")
print()

for doc_dir in sorted(OUTPUT_DIR.iterdir()):
    if doc_dir.is_dir():
        print(f"📁 {doc_dir.name}/")
        files = sorted(doc_dir.iterdir())
        for file in files:
            if file.is_file():
                size_kb = file.stat().st_size / 1024
                print(f"   • {file.name} ({size_kb:.1f} KB)")
        print()

print("=" * 80)
if len(successful) == len(test_docs):
    print("🎉 ALL TESTS PASSED - MVP COMPLETE")
else:
    print(f"⚠️  {len(failed) + len(skipped)} TESTS FAILED OR SKIPPED")
print("=" * 80)
print()
print(f"✨ All outputs saved to: {OUTPUT_DIR.absolute()}")
