"""DocTags exporter for VLM-compatible format."""

from typing import Any

from vertector_data_ingestion.exporters.base import BaseExporter


class DocTagsExporter(BaseExporter):
    """Export documents to DocTags format for VLM processing."""

    def __init__(self):
        """Initialize DocTags exporter."""
        pass

    def export(self, document: Any) -> str:
        """
        Export document to DocTags format.

        DocTags preserves structural tokens that are important for
        VLM (Vision-Language Model) processing, especially when using
        Granite-Docling-258M.

        Args:
            document: DoclingDocument to export

        Returns:
            DocTags string with structural markup
        """
        # Use document's built-in export method
        if hasattr(document, 'export_to_document_tokens'):
            return document.export_to_document_tokens()
        else:
            # Fallback: use JSON export
            return str(document)
