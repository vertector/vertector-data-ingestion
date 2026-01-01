"""Universal converter orchestrator for document processing."""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional, Tuple, Union

from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
    WordFormatOption,
    PowerpointFormatOption,
    ExcelFormatOption,
    HTMLFormatOption,
    ImageFormatOption,
)
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import ConvertPipelineOptions
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from vertector_data_ingestion.models.config import ConverterConfig, ExportFormat, PipelineType
from vertector_data_ingestion.models.document import (
    DoclingDocumentWrapper,
    DocumentMetadata,
)
from vertector_data_ingestion.core.pipeline_router import PipelineRouter
from vertector_data_ingestion.core.hardware_detector import HardwareDetector


class UniversalConverter:
    """Universal document converter with intelligent pipeline routing."""

    def __init__(self, config: Optional[ConverterConfig] = None):
        """
        Initialize universal converter.

        Args:
            config: Converter configuration (uses defaults if None)
        """
        self.config = config or ConverterConfig()
        self.hardware_config = HardwareDetector.detect()
        self.router = PipelineRouter(self.config)

        logger.info(f"Initialized UniversalConverter on {self.hardware_config.device_type.value}")

        # Model management
        if self.config.auto_download_models:
            self._ensure_models_available()

        # Cache management
        self.cache = None
        if self.config.enable_cache:
            from vertector_data_ingestion.utils.cache import ConversionCache

            self.cache = ConversionCache(
                cache_dir=self.config.cache_dir,
                enabled=True,
            )
            logger.info(f"Cache enabled at {self.config.cache_dir}")

    def _ensure_models_available(self):
        """
        Ensure required models are downloaded.

        Downloads models to cache directory if not present.
        """
        logger.info("Checking model availability...")

        # Create model cache directory
        self.config.model_cache_dir.mkdir(parents=True, exist_ok=True)

        # Models will be downloaded automatically by Docling on first use
        # This is just a placeholder for explicit model management if needed
        logger.debug(f"Models will be cached to: {self.config.model_cache_dir}")

    def _build_format_options(self, pipeline_type: Optional[PipelineType] = None):
        """
        Build format-specific options for all supported document types.

        Args:
            pipeline_type: Pipeline type for PDF processing

        Returns:
            Dictionary of format options for DocumentConverter
        """
        # Build PDF pipeline options (Classic or VLM)
        pdf_pipeline_options = self.router.build_pipeline_options(
            Path("dummy.pdf"), pipeline_type
        )

        # Import VlmPipeline for VLM mode
        from docling.pipeline.vlm_pipeline import VlmPipeline
        from docling.datamodel.pipeline_options import VlmPipelineOptions

        # Determine if we're using VLM pipeline
        is_vlm_pipeline = isinstance(pdf_pipeline_options, VlmPipelineOptions)

        # Build enrichment options for non-PDF formats
        # Enable AI-driven image descriptions based on hardware support
        enrich_options = ConvertPipelineOptions()

        # MPS doesn't support BFloat16, which is required by VLM models
        # Only enable picture descriptions on CUDA or CPU
        from vertector_data_ingestion.core.hardware_detector import HardwareType
        supports_vlm = self.hardware_config.device_type in [HardwareType.CUDA, HardwareType.CPU]

        if self.config.vlm.enable_picture_description and supports_vlm:
            enrich_options.do_picture_description = True
            enrich_options.do_picture_classification = True
            logger.debug("AI image enrichment enabled (hardware supports BFloat16)")
        else:
            if self.hardware_config.device_type == HardwareType.MPS:
                logger.debug("AI image enrichment disabled (MPS doesn't support BFloat16)")
            else:
                logger.debug("AI image enrichment disabled by configuration")

        # Create format-specific options
        # Use VlmPipeline class when VLM mode is enabled
        if is_vlm_pipeline:
            format_options = {
                InputFormat.PDF: PdfFormatOption(
                    pipeline_cls=VlmPipeline,
                    pipeline_options=pdf_pipeline_options
                ),
                InputFormat.DOCX: WordFormatOption(pipeline_options=enrich_options),
                InputFormat.PPTX: PowerpointFormatOption(pipeline_options=enrich_options),
                InputFormat.XLSX: ExcelFormatOption(pipeline_options=enrich_options),
                InputFormat.HTML: HTMLFormatOption(pipeline_options=enrich_options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pdf_pipeline_options),
            }
        else:
            format_options = {
                InputFormat.PDF: PdfFormatOption(pipeline_options=pdf_pipeline_options),
                InputFormat.DOCX: WordFormatOption(pipeline_options=enrich_options),
                InputFormat.PPTX: PowerpointFormatOption(pipeline_options=enrich_options),
                InputFormat.XLSX: ExcelFormatOption(pipeline_options=enrich_options),
                InputFormat.HTML: HTMLFormatOption(pipeline_options=enrich_options),
                InputFormat.IMAGE: ImageFormatOption(pipeline_options=pdf_pipeline_options),
            }

        return format_options

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _convert_with_retry(
        self, source: Path, pipeline_type: Optional[PipelineType] = None
    ) -> DoclingDocumentWrapper:
        """
        Convert document with automatic retry on transient failures.

        Args:
            source: Path to source document
            pipeline_type: Pipeline type (auto-detect if None)

        Returns:
            DoclingDocumentWrapper with converted document

        Raises:
            Exception: If conversion fails after retries
        """
        start_time = time.time()

        # Get pipeline type for metadata
        if pipeline_type is None:
            pipeline_type = self.router.determine_pipeline(source)

        # Build format options for all supported formats
        format_options = self._build_format_options(pipeline_type)

        # Create converter with format-specific options
        converter = DocumentConverter(
            format_options=format_options
        )

        # Convert document
        logger.info(f"Converting {source.name} with {pipeline_type.value} pipeline")
        result = converter.convert(source)

        # Extract DoclingDocument
        doc = result.document

        # Get page count
        num_pages = len(doc.pages) if hasattr(doc, "pages") else 1

        # Create metadata
        metadata = DocumentMetadata(
            source_path=source,
            num_pages=num_pages,
            title=doc.name if hasattr(doc, "name") else source.stem,
            format=source.suffix[1:],  # Remove leading dot
            pipeline_type=pipeline_type.value,
            processing_time=time.time() - start_time,
        )

        logger.info(
            f"Converted {source.name} in {metadata.processing_time:.2f}s "
            f"({num_pages} pages, {num_pages/metadata.processing_time:.1f} pages/sec)"
        )

        return DoclingDocumentWrapper(doc=doc, metadata=metadata)

    def convert(
        self,
        source: Union[Path, List[Path]],
        use_vlm: Optional[bool] = None,
        parallel: bool = True,
    ) -> Union[DoclingDocumentWrapper, List[DoclingDocumentWrapper]]:
        """
        Convert single document or batch of documents.

        Automatically detects whether source is a single file or list of files
        and processes accordingly.

        Args:
            source: Single Path or List of Paths to convert
            use_vlm: Force VLM pipeline (True) or Classic (False), or auto-detect (None)
            parallel: Use parallel processing for batches (ignored for single files)

        Returns:
            Single DoclingDocumentWrapper for single file, or
            List of DoclingDocumentWrapper for batch (only successful conversions)

        Examples:
            >>> converter = UniversalConverter()
            >>>
            >>> # Single file
            >>> doc = converter.convert(Path("document.pdf"))
            >>> print(doc.metadata.num_pages)
            >>>
            >>> # Batch processing
            >>> docs = converter.convert([
            ...     Path("doc1.pdf"),
            ...     Path("doc2.docx"),
            ...     Path("doc3.pptx")
            ... ])
            >>> for doc in docs:
            ...     print(f"{doc.metadata.source_path.name}: {doc.metadata.num_pages} pages")
        """
        if isinstance(source, (str, Path)):
            # Single file
            return self.convert_single(Path(source), use_vlm=use_vlm)
        elif isinstance(source, list):
            # Batch
            results = self.convert_batch(
                [Path(s) if isinstance(s, str) else s for s in source],
                parallel=parallel,
                fail_fast=self.config.fail_fast
            )
            # Extract successful conversions
            successful_docs = [doc for _, doc, err in results if doc is not None]

            # Log any failures
            failed_count = len(results) - len(successful_docs)
            if failed_count > 0:
                logger.warning(
                    f"Batch conversion: {len(successful_docs)} successful, "
                    f"{failed_count} failed"
                )

            return successful_docs
        else:
            raise TypeError(
                f"source must be Path or List[Path], got {type(source).__name__}"
            )

    def convert_single(
        self, source: Path, use_vlm: Optional[bool] = None
    ) -> DoclingDocumentWrapper:
        """
        Convert a single document.

        Args:
            source: Path to source document
            use_vlm: Force VLM pipeline (True) or Classic (False), or auto-detect (None)

        Returns:
            DoclingDocumentWrapper with converted document
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(source, "internal")
            if cached:
                logger.info(f"Using cached conversion for {source.name}")
                # TODO: Deserialize from cache
                # For now, skip cache and reconvert

        # Determine pipeline type
        if use_vlm is not None:
            pipeline_type = PipelineType.VLM if use_vlm else PipelineType.CLASSIC
        else:
            pipeline_type = None  # Auto-detect

        # Convert with retry
        doc_wrapper = self._convert_with_retry(source, pipeline_type)

        # Cache result
        if self.cache:
            # TODO: Serialize to cache
            pass

        return doc_wrapper

    def convert_batch(
        self,
        sources: List[Path],
        parallel: bool = True,
        fail_fast: bool = None,
    ) -> List[Tuple[Path, Optional[DoclingDocumentWrapper], Optional[Exception]]]:
        """
        Convert multiple documents.

        Args:
            sources: List of source document paths
            parallel: Use parallel processing
            fail_fast: Stop on first error (uses config default if None)

        Returns:
            List of (source_path, document, error) tuples
        """
        if fail_fast is None:
            fail_fast = self.config.fail_fast

        results = []

        if parallel and len(sources) > 1:
            # Parallel processing
            logger.info(f"Converting {len(sources)} documents in parallel")

            with ThreadPoolExecutor(
                max_workers=self.config.batch_processing_workers
            ) as executor:
                # Submit all tasks
                future_to_source = {
                    executor.submit(self.convert_single, source): source
                    for source in sources
                }

                # Collect results
                for future in as_completed(future_to_source):
                    source = future_to_source[future]
                    try:
                        doc = future.result()
                        results.append((source, doc, None))
                    except Exception as e:
                        logger.error(f"Failed to convert {source.name}: {e}")
                        results.append((source, None, e))
                        if fail_fast:
                            # Cancel remaining tasks
                            for f in future_to_source:
                                f.cancel()
                            raise
        else:
            # Sequential processing
            logger.info(f"Converting {len(sources)} documents sequentially")

            for source in sources:
                try:
                    doc = self.convert_single(source)
                    results.append((source, doc, None))
                except Exception as e:
                    logger.error(f"Failed to convert {source.name}: {e}")
                    results.append((source, None, e))
                    if fail_fast:
                        raise

        # Log summary
        successful = sum(1 for _, doc, _ in results if doc is not None)
        failed = len(results) - successful
        logger.info(
            f"Batch conversion complete: {successful} successful, {failed} failed"
        )

        return results

    def export(
        self, doc_wrapper: DoclingDocumentWrapper, format: ExportFormat
    ) -> str:
        """
        Export document to specified format.

        Args:
            doc_wrapper: Document wrapper
            format: Export format

        Returns:
            Exported document as string
        """
        if format == ExportFormat.MARKDOWN:
            return doc_wrapper.to_markdown()
        elif format == ExportFormat.JSON:
            import json

            return json.dumps(doc_wrapper.to_json(), indent=2)
        elif format == ExportFormat.DOCTAGS:
            return doc_wrapper.to_doctags()
        else:
            raise ValueError(f"Unsupported export format: {format}")

    def convert_and_export(
        self,
        source: Path,
        output_name: Optional[str] = None,
        format: ExportFormat = ExportFormat.MARKDOWN,
        use_vlm: Optional[bool] = None,
    ) -> Path:
        """
        Convert document and export to file.

        Args:
            source: Source document path
            output_name: Output filename (just the name, not full path).
                        If None, uses source filename with format extension.
                        The file will be saved in the configured output_dir.
            format: Export format
            use_vlm: Force VLM pipeline

        Returns:
            Path to output file

        Examples:
            >>> converter = UniversalConverter()
            >>> # Exports to {output_dir}/document.md
            >>> converter.convert_and_export("document.pdf", "document.md")
            >>> # Auto-generates name: {output_dir}/document.md
            >>> converter.convert_and_export("document.pdf")
        """
        # Convert document
        doc_wrapper = self.convert_single(source, use_vlm=use_vlm)

        # Export
        content = self.export(doc_wrapper, format)

        # Determine output filename
        if output_name is None:
            # Auto-generate from source filename
            ext_map = {
                ExportFormat.MARKDOWN: ".md",
                ExportFormat.JSON: ".json",
                ExportFormat.DOCTAGS: ".json",
            }
            ext = ext_map.get(format, ".txt")
            output_name = source.stem + ext

        # Build full output path using configured output_dir
        output_path = self.config.output_dir / output_name

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        output_path.write_text(content, encoding="utf-8")

        logger.info(f"Exported to {output_path}")

        return output_path
