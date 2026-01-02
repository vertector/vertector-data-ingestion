"""Markdown exporter for documents."""

from typing import Any

from vertector_data_ingestion.exporters.base import BaseExporter


class MarkdownExporter(BaseExporter):
    """Export documents to Markdown format."""

    def __init__(self, exclude_furniture: bool = True):
        """
        Initialize markdown exporter.

        Args:
            exclude_furniture: If True, exclude headers/footers/page numbers
        """
        self.exclude_furniture = exclude_furniture

    def export(self, document: Any) -> str:
        """
        Export document to Markdown.

        Args:
            document: DoclingDocument to export

        Returns:
            Markdown string
        """
        # Use document's built-in export method
        if hasattr(document, "export_to_markdown"):
            return document.export_to_markdown()
        elif hasattr(document, "to_markdown"):
            return document.to_markdown()
        else:
            # Fallback: convert to string
            return str(document)
