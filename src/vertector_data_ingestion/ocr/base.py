"""Base OCR plugin interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class OcrResult:
    """OCR result for a text region."""

    text: str
    confidence: float
    bbox: tuple  # (x1, y1, x2, y2)


class OcrPlugin(ABC):
    """Abstract base class for OCR plugins."""

    @abstractmethod
    def initialize(self, languages: List[str], use_gpu: bool) -> None:
        """
        Initialize OCR engine.

        Args:
            languages: List of language codes (e.g., ['en', 'fr'])
            use_gpu: Whether to use GPU acceleration
        """
        pass

    @abstractmethod
    def extract_text(
        self, image: np.ndarray, confidence_threshold: float = 0.5
    ) -> List[OcrResult]:
        """
        Extract text from image.

        Args:
            image: Image as numpy array
            confidence_threshold: Minimum confidence for text detection

        Returns:
            List of OCR results
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if OCR engine is available.

        Returns:
            True if engine can be used
        """
        pass
