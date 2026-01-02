"""
Compare OCR engines: EasyOCR vs OCRMac (macOS native).

Tests both OCR engines on the same document and compares:
1. Accuracy of text extraction
2. Processing speed
3. Text completeness
"""

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vertector_data_ingestion.core.universal_converter import UniversalConverter
from vertector_data_ingestion.models.config import (
    ConverterConfig,
    ExportFormat,
    LocalMpsConfig,
    OcrEngine,
)

print("=" * 80)
print("OCR ENGINE COMPARISON TEST")
print("=" * 80)
print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# Test document
test_pdf = Path("test_documents/ocr_test.pdf")

if not test_pdf.exists():
    print(f"❌ Test file not found: {test_pdf}")
    print("Please run test_ocr.py first to create the test PDF")
    sys.exit(1)

results = {}

# Test 1: EasyOCR
print("=" * 80)
print("TEST 1: EasyOCR Engine")
print("=" * 80)
print()

config_easyocr = ConverterConfig()
config_easyocr.ocr.engine = OcrEngine.EASYOCR

print(f"Configuration: {config_easyocr.ocr.engine.value}")
converter_easyocr = UniversalConverter(config_easyocr)

try:
    doc_easyocr = converter_easyocr.convert_single(test_pdf)
    markdown_easyocr = converter_easyocr.export(doc_easyocr, ExportFormat.MARKDOWN)

    results["easyocr"] = {
        "time": doc_easyocr.metadata.processing_time,
        "text": markdown_easyocr,
        "chars": len(markdown_easyocr),
        "words": len(markdown_easyocr.split()),
        "pages": doc_easyocr.metadata.num_pages,
    }

    print(f"✅ Processing time: {results['easyocr']['time']:.2f}s")
    print(f"   Characters extracted: {results['easyocr']['chars']:,}")
    print(f"   Words extracted: {results['easyocr']['words']:,}")
    print()
    print("Extracted text:")
    print("-" * 80)
    print(markdown_easyocr)
    print()

except Exception as e:
    print(f"❌ EasyOCR failed: {e}")
    import traceback

    traceback.print_exc()

# Test 2: OCRMac (macOS native)
print("=" * 80)
print("TEST 2: OCRMac Engine (macOS Native)")
print("=" * 80)
print()

config_ocrmac = LocalMpsConfig()  # Uses OCRMAC by default

print(f"Configuration: {config_ocrmac.ocr.engine.value}")
converter_ocrmac = UniversalConverter(config_ocrmac)

try:
    doc_ocrmac = converter_ocrmac.convert_single(test_pdf)
    markdown_ocrmac = converter_ocrmac.export(doc_ocrmac, ExportFormat.MARKDOWN)

    results["ocrmac"] = {
        "time": doc_ocrmac.metadata.processing_time,
        "text": markdown_ocrmac,
        "chars": len(markdown_ocrmac),
        "words": len(markdown_ocrmac.split()),
        "pages": doc_ocrmac.metadata.num_pages,
    }

    print(f"✅ Processing time: {results['ocrmac']['time']:.2f}s")
    print(f"   Characters extracted: {results['ocrmac']['chars']:,}")
    print(f"   Words extracted: {results['ocrmac']['words']:,}")
    print()
    print("Extracted text:")
    print("-" * 80)
    print(markdown_ocrmac)
    print()

except Exception as e:
    print(f"❌ OCRMac failed: {e}")
    import traceback

    traceback.print_exc()

# Comparison
if "easyocr" in results and "ocrmac" in results:
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()

    print("Speed:")
    print(f"  EasyOCR:  {results['easyocr']['time']:.2f}s")
    print(f"  OCRMac:   {results['ocrmac']['time']:.2f}s")

    faster_engine = (
        "EasyOCR" if results["easyocr"]["time"] < results["ocrmac"]["time"] else "OCRMac"
    )
    speed_diff = abs(results["easyocr"]["time"] - results["ocrmac"]["time"])
    print(f"  Winner: {faster_engine} ({speed_diff:.2f}s faster)")
    print()

    print("Text Extraction:")
    print(f"  EasyOCR:  {results['easyocr']['chars']} chars, {results['easyocr']['words']} words")
    print(f"  OCRMac:   {results['ocrmac']['chars']} chars, {results['ocrmac']['words']} words")

    more_complete = (
        "EasyOCR" if results["easyocr"]["chars"] > results["ocrmac"]["chars"] else "OCRMac"
    )
    print(f"  More complete: {more_complete}")
    print()

    print("Recommendation:")
    if results["ocrmac"]["time"] < results["easyocr"]["time"] * 0.8:
        print("  ✅ OCRMac is significantly faster - recommended for macOS")
    elif results["easyocr"]["chars"] > results["ocrmac"]["chars"] * 1.2:
        print("  ✅ EasyOCR extracts more text - recommended for accuracy")
    else:
        print("  ✅ Both engines perform similarly - use LocalMpsConfig (OCRMac) on macOS")
    print()

print("=" * 80)
print("✅ OCR COMPARISON COMPLETE")
print("=" * 80)
