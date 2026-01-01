"""Base exporter interface."""

from abc import ABC, abstractmethod
from typing import Any


class BaseExporter(ABC):
    """Abstract base class for document exporters."""

    @abstractmethod
    def export(self, document: Any) -> str:
        """
        Export document to string format.

        Args:
            document: DoclingDocument to export

        Returns:
            Exported document as string
        """
        pass
