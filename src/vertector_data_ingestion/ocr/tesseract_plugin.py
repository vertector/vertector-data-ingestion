"""Tesseract OCR plugin implementation."""

from typing import List

import numpy as np
from loguru import logger

from vertector_data_ingestion.ocr.base import OcrPlugin, OcrResult


class TesseractPlugin(OcrPlugin):
    """Tesseract OCR plugin for CPU-friendly OCR."""

    def __init__(self):
        """Initialize Tesseract plugin."""
        self.languages = []
        self.initialized = False

    def initialize(self, languages: List[str], use_gpu: bool) -> None:
        """
        Initialize Tesseract engine.

        Args:
            languages: List of language codes
            use_gpu: Ignored (Tesseract is CPU-only)
        """
        try:
            import pytesseract

            # Test if tesseract is installed
            pytesseract.get_tesseract_version()

            self.languages = languages
            self.initialized = True

            if use_gpu:
                logger.warning("Tesseract does not support GPU acceleration")

            logger.info(f"Tesseract initialized with languages: {languages}")

        except ImportError:
            logger.error("pytesseract not installed. Install with: uv add pytesseract")
            raise
        except Exception as e:
            logger.error(
                f"Tesseract not found. Install tesseract-ocr: "
                f"brew install tesseract (macOS) or apt-get install tesseract-ocr (Linux)"
            )
            raise

    def extract_text(
        self, image: np.ndarray, confidence_threshold: float = 0.5
    ) -> List[OcrResult]:
        """
        Extract text from image using Tesseract.

        Args:
            image: Image as numpy array
            confidence_threshold: Minimum confidence for text detection

        Returns:
            List of OCR results
        """
        if not self.initialized:
            raise RuntimeError("Tesseract not initialized. Call initialize() first.")

        try:
            import pytesseract
            from PIL import Image

            # Convert numpy array to PIL Image
            pil_image = Image.fromarray(image)

            # Get detailed output with bounding boxes
            lang_str = "+".join(self.languages)
            data = pytesseract.image_to_data(
                pil_image,
                lang=lang_str,
                output_type=pytesseract.Output.DICT,
            )

            # Parse results
            ocr_results = []
            n_boxes = len(data["text"])

            for i in range(n_boxes):
                text = data["text"][i]
                conf = int(data["conf"][i])

                # Filter by confidence (Tesseract uses 0-100 scale)
                if conf >= confidence_threshold * 100 and text.strip():
                    x, y, w, h = (
                        data["left"][i],
                        data["top"][i],
                        data["width"][i],
                        data["height"][i],
                    )
                    bbox = (x, y, x + w, y + h)

                    ocr_results.append(
                        OcrResult(
                            text=text,
                            confidence=conf / 100.0,  # Normalize to 0-1
                            bbox=bbox,
                        )
                    )

            return ocr_results

        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            raise

    def is_available(self) -> bool:
        """
        Check if Tesseract is available.

        Returns:
            True if Tesseract can be used
        """
        try:
            import pytesseract

            pytesseract.get_tesseract_version()
            return True
        except Exception:
            return False
