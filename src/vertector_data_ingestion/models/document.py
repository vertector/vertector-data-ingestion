"""Document models and wrappers for Docling integration."""

from pathlib import Path
from typing import Any

from pydantic import BaseModel


class BoundingBox(BaseModel):
    """Bounding box coordinates for visual grounding."""

    l: float  # left
    t: float  # top
    r: float  # right
    b: float  # bottom
    page: int


class DocumentMetadata(BaseModel):
    """Metadata extracted from document."""

    source_path: Path
    num_pages: int
    title: str | None = None
    author: str | None = None
    creation_date: str | None = None
    modification_date: str | None = None
    format: str
    pipeline_type: str  # classic or vlm
    processing_time: float | None = None


class TableCell(BaseModel):
    """Table cell with content and position."""

    row: int
    col: int
    content: str
    rowspan: int = 1
    colspan: int = 1
    bbox: BoundingBox | None = None


class Table(BaseModel):
    """Table structure with cells and metadata."""

    cells: list[TableCell]
    num_rows: int
    num_cols: int
    caption: str | None = None
    page_no: int
    bbox: BoundingBox | None = None


class DoclingDocumentWrapper(BaseModel):
    """Wrapper around DoclingDocument for easier manipulation."""

    doc: Any  # CoreDoclingDocument - using Any to avoid serialization issues
    metadata: DocumentMetadata

    class Config:
        arbitrary_types_allowed = True

    def to_markdown(self, exclude_furniture: bool = True) -> str:
        """
        Export document to markdown.

        Args:
            exclude_furniture: If True, exclude headers/footers/page numbers

        Returns:
            Markdown string
        """
        # Use built-in export method from DoclingDocument
        return self.doc.export_to_markdown()

    def to_json(self, lossless: bool = True) -> dict[str, Any]:
        """
        Export document to JSON.

        Args:
            lossless: If True, include all metadata (bboxes, provenance)

        Returns:
            JSON dict
        """
        # Use built-in export method from DoclingDocument
        return self.doc.export_to_dict()

    def to_doctags(self) -> str:
        """
        Export document to DocTags format for VLM.

        Returns:
            DocTags string
        """
        # Use built-in export method from DoclingDocument
        return self.doc.export_to_doctags()

    def extract_tables(self) -> list[Table]:
        """
        Extract all tables from document.

        Returns:
            List of Table objects
        """
        tables = []

        # Iterate through document items
        for item in self.doc.body:
            if item.obj_type == "table":
                # Extract table structure
                table_data = item.table_data
                if table_data:
                    cells = []
                    for cell in table_data.cells:
                        cells.append(
                            TableCell(
                                row=cell.row,
                                col=cell.col,
                                content=cell.text,
                                rowspan=cell.rowspan,
                                colspan=cell.colspan,
                                bbox=BoundingBox(**cell.bbox) if cell.bbox else None,
                            )
                        )

                    tables.append(
                        Table(
                            cells=cells,
                            num_rows=table_data.num_rows,
                            num_cols=table_data.num_cols,
                            caption=getattr(item, "caption", None),
                            page_no=item.prov[0].page_no if item.prov else 0,
                            bbox=BoundingBox(**item.prov[0].bbox) if item.prov else None,
                        )
                    )

        return tables

    def get_text(self, exclude_furniture: bool = True) -> str:
        """
        Get plain text from document.

        Args:
            exclude_furniture: If True, exclude headers/footers/page numbers

        Returns:
            Plain text string
        """
        text_parts = []

        items = self.doc.body if exclude_furniture else list(self.doc.iterate_items())

        for item in items:
            if hasattr(item, "text") and item.text:
                text_parts.append(item.text)

        return "\n\n".join(text_parts)

    def get_page_count(self) -> int:
        """Get total number of pages in document."""
        return self.metadata.num_pages

    def get_content_by_page(self, page_no: int) -> list[dict[str, Any]]:
        """
        Get all content items for a specific page.

        Args:
            page_no: Page number (1-indexed)

        Returns:
            List of content items with text and metadata
        """
        page_content = []

        for item in self.doc.iterate_items():
            if item.prov and any(p.page_no == page_no for p in item.prov):
                page_content.append(
                    {
                        "type": item.obj_type,
                        "text": getattr(item, "text", None),
                        "bbox": item.prov[0].bbox if item.prov else None,
                    }
                )

        return page_content
