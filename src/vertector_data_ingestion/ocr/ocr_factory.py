"""Factory for creating OCR plugins."""

from loguru import logger

from vertector_data_ingestion.models.config import OcrEngine
from vertector_data_ingestion.ocr.base import OcrPlugin
from vertector_data_ingestion.ocr.easyocr_plugin import EasyOcrPlugin
from vertector_data_ingestion.ocr.ocrmac_plugin import OcrMacPlugin
from vertector_data_ingestion.ocr.tesseract_plugin import TesseractPlugin


class OcrFactory:
    """Factory for creating OCR plugins."""

    _plugins = {
        OcrEngine.EASYOCR: EasyOcrPlugin,
        OcrEngine.TESSERACT: TesseractPlugin,
        OcrEngine.OCRMAC: OcrMacPlugin,
    }

    @classmethod
    def create(cls, engine: OcrEngine, languages: list, use_gpu: bool) -> OcrPlugin:
        """
        Create and initialize OCR plugin.

        Args:
            engine: OCR engine type
            languages: List of language codes
            use_gpu: Whether to use GPU acceleration

        Returns:
            Initialized OCR plugin

        Raises:
            ValueError: If engine is not supported
            RuntimeError: If plugin is not available
        """
        plugin_class = cls._plugins.get(engine)
        if not plugin_class:
            raise ValueError(f"Unsupported OCR engine: {engine}")

        # Create plugin
        plugin = plugin_class()

        # Check availability
        if not plugin.is_available():
            logger.warning(f"{engine.value} is not available, trying fallback")
            return cls._create_fallback(languages, use_gpu)

        # Initialize plugin
        try:
            plugin.initialize(languages, use_gpu)
            return plugin
        except Exception as e:
            logger.error(f"Failed to initialize {engine.value}: {e}")
            return cls._create_fallback(languages, use_gpu)

    @classmethod
    def _create_fallback(cls, languages: list, use_gpu: bool) -> OcrPlugin:
        """
        Create fallback OCR plugin.

        Tries plugins in order: EasyOCR -> Tesseract

        Args:
            languages: List of language codes
            use_gpu: Whether to use GPU

        Returns:
            First available plugin

        Raises:
            RuntimeError: If no plugins are available
        """
        # Try EasyOCR first
        try:
            plugin = EasyOcrPlugin()
            if plugin.is_available():
                plugin.initialize(languages, use_gpu)
                logger.info("Using EasyOCR as fallback")
                return plugin
        except Exception:
            pass

        # Try Tesseract
        try:
            plugin = TesseractPlugin()
            if plugin.is_available():
                plugin.initialize(languages, False)  # Tesseract is CPU-only
                logger.info("Using Tesseract as fallback")
                return plugin
        except Exception:
            pass

        raise RuntimeError("No OCR engines available. Please install easyocr or tesseract.")
