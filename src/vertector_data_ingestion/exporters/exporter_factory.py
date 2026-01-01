"""Factory for creating exporters."""

from vertector_data_ingestion.models.config import ExportFormat
from vertector_data_ingestion.exporters.base import BaseExporter
from vertector_data_ingestion.exporters.markdown_exporter import MarkdownExporter
from vertector_data_ingestion.exporters.json_exporter import JsonExporter
from vertector_data_ingestion.exporters.doctags_exporter import DocTagsExporter


class ExporterFactory:
    """Factory for creating document exporters."""

    _exporters = {
        ExportFormat.MARKDOWN: MarkdownExporter,
        ExportFormat.JSON: JsonExporter,
        ExportFormat.DOCTAGS: DocTagsExporter,
    }

    @classmethod
    def create(cls, format: ExportFormat, **kwargs) -> BaseExporter:
        """
        Create exporter for specified format.

        Args:
            format: Export format
            **kwargs: Additional arguments for exporter

        Returns:
            Exporter instance

        Raises:
            ValueError: If format is not supported
        """
        exporter_class = cls._exporters.get(format)
        if not exporter_class:
            raise ValueError(f"Unsupported export format: {format}")

        return exporter_class(**kwargs)
