"""OcrMac plugin for native macOS OCR using Vision framework."""

import platform
from typing import List

import numpy as np
from loguru import logger

from vertector_data_ingestion.ocr.base import OcrPlugin, OcrResult


class OcrMacPlugin(OcrPlugin):
    """Native macOS OCR plugin using Vision framework."""

    def __init__(self):
        """Initialize OcrMac plugin."""
        self.languages = []
        self.initialized = False
        self.is_macos = platform.system() == "Darwin"

    def initialize(self, languages: List[str], use_gpu: bool) -> None:
        """
        Initialize OcrMac engine.

        Args:
            languages: List of language codes (e.g., ['en', 'fr'])
            use_gpu: Ignored (Vision framework handles acceleration automatically)
        """
        if not self.is_macos:
            raise RuntimeError("OcrMac is only available on macOS")

        try:
            # Try importing Docling's OcrMac
            from docling.datamodel.pipeline_options import OcrMacOptions

            self.languages = languages
            self.initialized = True

            logger.info(f"OcrMac initialized with languages: {languages}")
            logger.info("Using native macOS Vision framework for OCR")

        except ImportError:
            logger.error(
                "OcrMac not available. This feature requires Docling with macOS support."
            )
            raise

    def extract_text(
        self, image: np.ndarray, confidence_threshold: float = 0.5
    ) -> List[OcrResult]:
        """
        Extract text from image using macOS Vision framework.

        Args:
            image: Image as numpy array
            confidence_threshold: Minimum confidence for text detection

        Returns:
            List of OCR results

        Note:
            This is a placeholder. In practice, OcrMac is integrated directly
            into Docling's pipeline and doesn't expose a standalone API.
            This plugin exists primarily for consistency with the plugin architecture.
        """
        if not self.initialized:
            raise RuntimeError("OcrMac not initialized. Call initialize() first.")

        # OcrMac is typically used through Docling's pipeline integration
        # This standalone method would require direct Vision framework access
        raise NotImplementedError(
            "OcrMac extraction should be used through Docling's pipeline integration"
        )

    def is_available(self) -> bool:
        """
        Check if OcrMac is available.

        Returns:
            True if running on macOS and OcrMac can be imported
        """
        if not self.is_macos:
            return False

        try:
            from docling.datamodel.pipeline_options import OcrMacOptions

            return True
        except ImportError:
            return False
