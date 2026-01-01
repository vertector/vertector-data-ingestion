"""Caching utilities for conversion results."""

import hashlib
import json
from pathlib import Path
from typing import Optional

from loguru import logger


class ConversionCache:
    """File-based cache for document conversion results."""

    def __init__(self, cache_dir: Path, enabled: bool = False):
        """
        Initialize cache.

        Args:
            cache_dir: Directory for cache storage
            enabled: Whether caching is enabled
        """
        self.cache_dir = cache_dir
        self.enabled = enabled

        if enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Cache initialized at {cache_dir}")

    def _compute_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of file.

        Args:
            file_path: Path to file

        Returns:
            Hex digest of file hash
        """
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read in chunks to handle large files
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def get(self, source: Path, export_format: str) -> Optional[str]:
        """
        Retrieve cached conversion result.

        Args:
            source: Source file path
            export_format: Export format (e.g., "markdown", "json")

        Returns:
            Cached content if found, None otherwise
        """
        if not self.enabled:
            return None

        try:
            file_hash = self._compute_hash(source)
            cache_file = self.cache_dir / f"{file_hash}_{export_format}.cache"

            if cache_file.exists():
                content = cache_file.read_text(encoding="utf-8")
                logger.debug(f"Cache hit for {source.name}")
                return content

        except Exception as e:
            logger.warning(f"Cache read error: {e}")

        return None

    def set(self, source: Path, export_format: str, result: str):
        """
        Store conversion result in cache.

        Args:
            source: Source file path
            export_format: Export format
            result: Conversion result to cache
        """
        if not self.enabled:
            return

        try:
            file_hash = self._compute_hash(source)
            cache_file = self.cache_dir / f"{file_hash}_{export_format}.cache"

            cache_file.write_text(result, encoding="utf-8")
            logger.debug(f"Cached result for {source.name}")

        except Exception as e:
            logger.warning(f"Cache write error: {e}")

    def clear(self):
        """Clear all cached files."""
        if not self.enabled:
            return

        try:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink()
            logger.info("Cache cleared")
        except Exception as e:
            logger.error(f"Cache clear error: {e}")

    def get_size(self) -> int:
        """
        Get total cache size in bytes.

        Returns:
            Total size of all cache files
        """
        if not self.enabled:
            return 0

        total_size = 0
        for cache_file in self.cache_dir.glob("*.cache"):
            total_size += cache_file.stat().st_size

        return total_size

    def get_stats(self) -> dict:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache stats
        """
        if not self.enabled:
            return {"enabled": False}

        cache_files = list(self.cache_dir.glob("*.cache"))

        return {
            "enabled": True,
            "cache_dir": str(self.cache_dir),
            "num_files": len(cache_files),
            "total_size_mb": self.get_size() / (1024 * 1024),
        }
