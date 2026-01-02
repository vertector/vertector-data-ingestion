"""MCP Server for Vertector Data Ingestion.

Provides Model Context Protocol (MCP) tools for document parsing,
chunking, and metadata enrichment.

Usage:
    # Run as MCP server
    vertector-data-ingestion-mcp

    # Or run programmatically
    python -m vertector_data_ingestion.mcp.server
"""

from .server import app, main, async_main

__all__ = ["app", "main", "async_main"]
