"""JSON exporter for lossless document representation."""

import json
from typing import Any

from vertector_data_ingestion.exporters.base import BaseExporter


class JsonExporter(BaseExporter):
    """Export documents to lossless JSON format."""

    def __init__(self, lossless: bool = True, indent: int = 2):
        """
        Initialize JSON exporter.

        Args:
            lossless: If True, include all metadata (bboxes, provenance)
            indent: JSON indentation spaces
        """
        self.lossless = lossless
        self.indent = indent

    def export(self, document: Any) -> str:
        """
        Export document to JSON.

        Args:
            document: DoclingDocument to export

        Returns:
            JSON string
        """
        # Convert DoclingDocument to dict
        doc_dict = document.model_dump(exclude_none=not self.lossless)

        # Serialize to JSON
        return json.dumps(doc_dict, indent=self.indent, ensure_ascii=False)

    def export_to_dict(self, document: Any) -> dict[str, Any]:
        """
        Export document to dictionary.

        Args:
            document: DoclingDocument to export

        Returns:
            Dictionary representation
        """
        return document.model_dump(exclude_none=not self.lossless)
