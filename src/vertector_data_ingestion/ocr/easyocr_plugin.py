"""EasyOCR plugin implementation."""

import numpy as np
from loguru import logger

from vertector_data_ingestion.ocr.base import OcrPlugin, OcrResult


class EasyOcrPlugin(OcrPlugin):
    """EasyOCR plugin for GPU-accelerated OCR."""

    def __init__(self):
        """Initialize EasyOCR plugin."""
        self.reader = None
        self.initialized = False

    def initialize(self, languages: list[str], use_gpu: bool) -> None:
        """
        Initialize EasyOCR engine.

        Args:
            languages: List of language codes
            use_gpu: Whether to use GPU acceleration
        """
        try:
            import easyocr

            self.reader = easyocr.Reader(
                lang_list=languages,
                gpu=use_gpu,
                verbose=False,
            )
            self.initialized = True
            logger.info(f"EasyOCR initialized with languages: {languages}, GPU: {use_gpu}")

        except ImportError:
            logger.error("EasyOCR not installed. Install with: uv add easyocr")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize EasyOCR: {e}")
            raise

    def extract_text(self, image: np.ndarray, confidence_threshold: float = 0.5) -> list[OcrResult]:
        """
        Extract text from image using EasyOCR.

        Args:
            image: Image as numpy array
            confidence_threshold: Minimum confidence for text detection

        Returns:
            List of OCR results
        """
        if not self.initialized or not self.reader:
            raise RuntimeError("EasyOCR not initialized. Call initialize() first.")

        try:
            # Run OCR
            results = self.reader.readtext(image)

            # Convert to OcrResult objects
            ocr_results = []
            for bbox, text, confidence in results:
                if confidence >= confidence_threshold:
                    # Convert bbox format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                    # to (x1, y1, x2, y2)
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    bbox_tuple = (
                        min(x_coords),
                        min(y_coords),
                        max(x_coords),
                        max(y_coords),
                    )

                    ocr_results.append(OcrResult(text=text, confidence=confidence, bbox=bbox_tuple))

            return ocr_results

        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            raise

    def is_available(self) -> bool:
        """
        Check if EasyOCR is available.

        Returns:
            True if EasyOCR can be imported
        """
        try:
            import easyocr

            return True
        except ImportError:
            return False
