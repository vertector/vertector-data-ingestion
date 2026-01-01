"""Pipeline routing logic for Classic vs VLM pipeline selection."""

from enum import Enum
from pathlib import Path
from typing import Optional

from docling.datamodel import vlm_model_specs
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
    EasyOcrOptions,
    TesseractOcrOptions,
    VlmPipelineOptions,
)
from docling.datamodel.pipeline_options_vlm_model import (
    InlineVlmOptions,
    ResponseFormat,
    InferenceFramework,
)
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice
from docling.pipeline.vlm_pipeline import VlmPipeline
from loguru import logger

from vertector_data_ingestion.models.config import (
    ConverterConfig,
    OcrEngine,
    PipelineType,
)
from vertector_data_ingestion.core.hardware_detector import HardwareDetector, HardwareType


class DocumentType(str, Enum):
    """Document type classification."""

    PDF_STANDARD = "pdf_standard"
    PDF_SCANNED = "pdf_scanned"
    PDF_TABLES = "pdf_tables"
    DOCX = "docx"
    PPTX = "pptx"
    XLSX = "xlsx"
    HTML = "html"
    UNKNOWN = "unknown"


class PipelineRouter:
    """Routes documents to appropriate pipeline (Classic or VLM)."""

    def __init__(self, config: ConverterConfig):
        """
        Initialize pipeline router.

        Args:
            config: Converter configuration
        """
        self.config = config
        self.hardware_config = HardwareDetector.detect()
        logger.info(f"Hardware detected: {self.hardware_config.device_type.value}")

    def determine_pipeline(
        self, source: Path, force_vlm: Optional[bool] = None
    ) -> PipelineType:
        """
        Determine which pipeline to use for a document.

        Args:
            source: Path to document
            force_vlm: Force VLM pipeline (True) or Classic (False), or auto-detect (None)

        Returns:
            Pipeline type to use
        """
        # Honor manual override
        if force_vlm is True:
            logger.info(f"Using VLM pipeline (forced) for {source.name}")
            return PipelineType.VLM
        elif force_vlm is False:
            logger.info(f"Using Classic pipeline (forced) for {source.name}")
            return PipelineType.CLASSIC

        # If user explicitly set default_pipeline to VLM (not auto), honor it
        # This allows VLM to be activated by configuration
        if self.config.default_pipeline == PipelineType.VLM and not self.config.auto_detect_pipeline:
            logger.info(f"Using VLM pipeline (configured default) for {source.name}")
            return PipelineType.VLM

        # If auto-detect is disabled, use configured default
        if not self.config.auto_detect_pipeline:
            logger.info(f"Using default pipeline: {self.config.default_pipeline.value}")
            return self.config.default_pipeline

        # Auto-detect document type for intelligent routing
        doc_type = self._detect_document_type(source)
        logger.debug(f"Detected document type: {doc_type.value}")

        # Route based on document type
        if doc_type == DocumentType.PDF_SCANNED:
            logger.info(f"Using VLM pipeline for scanned PDF: {source.name}")
            return PipelineType.VLM
        elif doc_type == DocumentType.PDF_TABLES:
            logger.info(f"Using Classic pipeline for PDF with tables: {source.name}")
            return PipelineType.CLASSIC
        elif doc_type in [DocumentType.PPTX]:
            logger.info(f"Using VLM pipeline for presentation: {source.name}")
            return PipelineType.VLM
        else:
            logger.info(f"Using Classic pipeline (default) for {source.name}")
            return PipelineType.CLASSIC

    def build_pipeline_options(
        self, source: Path, pipeline_type: Optional[PipelineType] = None
    ) -> PdfPipelineOptions:
        """
        Build PdfPipelineOptions for document conversion.

        Args:
            source: Path to document
            pipeline_type: Pipeline type (auto-detect if None)

        Returns:
            Configured PdfPipelineOptions
        """
        if pipeline_type is None:
            pipeline_type = self.determine_pipeline(source)

        if pipeline_type == PipelineType.VLM:
            return self._build_vlm_options()
        else:
            return self._build_classic_options()

    def _build_classic_options(self) -> PdfPipelineOptions:
        """Build options for Classic pipeline with TableFormer."""
        # Configure accelerator based on detected hardware
        device_map = {
            HardwareType.CUDA: AcceleratorDevice.CUDA,
            HardwareType.MPS: AcceleratorDevice.MPS,
            HardwareType.CPU: AcceleratorDevice.CPU,
        }
        accelerator_device = device_map.get(self.hardware_config.device_type, AcceleratorDevice.CPU)

        accelerator_options = AcceleratorOptions(
            num_threads=self.hardware_config.num_workers,
            device=accelerator_device,
        )

        # Create PdfPipelineOptions with TableFormer configuration
        # NOTE: Don't set artifacts_path to allow automatic model downloads
        options = PdfPipelineOptions(
            accelerator_options=accelerator_options,
        )

        # Enable and configure table structure extraction
        options.do_table_structure = True
        options.table_structure_options.mode = (
            TableFormerMode.ACCURATE if self.config.table.mode.value == "accurate"
            else TableFormerMode.FAST
        )
        options.table_structure_options.do_cell_matching = self.config.table.cell_matching

        # Configure OCR with proper model storage directory
        options.do_ocr = True
        options.ocr_options = self._get_ocr_options()

        logger.debug(
            f"Classic pipeline configured: device={accelerator_device.value}, "
            f"TableFormer={options.table_structure_options.mode.value}, "
            f"cell_matching={options.table_structure_options.do_cell_matching}, "
            f"OCR={options.do_ocr}"
        )
        return options

    def _build_vlm_options(self) -> VlmPipelineOptions:
        """
        Build options for VLM pipeline.

        VLM pipeline uses vision-language models for end-to-end document conversion,
        processing pages visually rather than relying on layout detection + OCR.
        """
        # Check if custom model is specified
        if self.config.vlm.custom_model_repo_id:
            # Use custom model from Hugging Face
            logger.info(f"Using custom VLM model: {self.config.vlm.custom_model_repo_id}")

            # Determine inference framework based on hardware and MLX setting
            if self.hardware_config.device_type == HardwareType.MPS and self.config.vlm.use_mlx:
                inference_framework = InferenceFramework.MLX
                logger.info("Using MLX framework for custom model on MPS")
            else:
                inference_framework = InferenceFramework.TRANSFORMERS
                logger.info("Using Transformers framework for custom model")

            # Use custom prompt or default engineered prompt
            default_prompt = self._get_default_vlm_prompt()

            # Create custom model options
            vlm_options = InlineVlmOptions(
                repo_id=self.config.vlm.custom_model_repo_id,
                prompt=self.config.vlm.custom_model_prompt or default_prompt,
                response_format=ResponseFormat.MARKDOWN,
                inference_framework=inference_framework,
            )

            options = VlmPipelineOptions(vlm_options=vlm_options)
            logger.debug(f"Custom VLM configured: {self.config.vlm.custom_model_repo_id}")
            return options

        # Use pre-configured models based on hardware and configuration
        import copy
        vlm_options = None

        # Check if user specified a preset model
        preset = self.config.vlm.preset_model

        if self.hardware_config.device_type == HardwareType.MPS and self.config.vlm.use_mlx:
            # macOS with MPS acceleration - select MLX model based on preset
            if preset == "qwen25-3b" and hasattr(vlm_model_specs, 'QWEN25_VL_3B_MLX'):
                vlm_options = copy.deepcopy(vlm_model_specs.QWEN25_VL_3B_MLX)
                logger.info("Using Qwen2.5-VL-3B-MLX (3B params, ~23.5s/page)")
            elif preset == "smoldocling-mlx" and hasattr(vlm_model_specs, 'SMOLDOCLING_MLX'):
                vlm_options = copy.deepcopy(vlm_model_specs.SMOLDOCLING_MLX)
                logger.info("Using SmolDocling-MLX (256M params, ~6.15s/page)")
            elif preset == "pixtral-12b" and hasattr(vlm_model_specs, 'PIXTRAL_12B_MLX'):
                vlm_options = copy.deepcopy(vlm_model_specs.PIXTRAL_12B_MLX)
                logger.info("Using Pixtral-12B-MLX (12B params, ~309s/page)")
            elif preset == "gemma3-12b" and hasattr(vlm_model_specs, 'GEMMA3_12B_MLX'):
                vlm_options = copy.deepcopy(vlm_model_specs.GEMMA3_12B_MLX)
                logger.info("Using Gemma-3-12B-MLX (12B params, ~378s/page)")
            elif preset == "granite-mlx" and hasattr(vlm_model_specs, 'GRANITEDOCLING_MLX'):
                vlm_options = copy.deepcopy(vlm_model_specs.GRANITEDOCLING_MLX)
                logger.info("Using Granite-Docling-MLX (258M params)")
            else:
                # Default to Granite-Docling MLX
                vlm_options = copy.deepcopy(vlm_model_specs.GRANITEDOCLING_MLX)
                logger.info("Using Granite-Docling-MLX (258M params, default)")

        elif self.hardware_config.device_type == HardwareType.MPS:
            # MPS without MLX - use Transformers models
            if preset == "granite-vision" and hasattr(vlm_model_specs, 'GRANITE_VISION_TRANSFORMERS'):
                vlm_options = copy.deepcopy(vlm_model_specs.GRANITE_VISION_TRANSFORMERS)
                logger.info("Using Granite-Vision-Transformers (2B params, ~105s/page)")
            elif preset == "smoldocling" and hasattr(vlm_model_specs, 'SMOLDOCLING_TRANSFORMERS'):
                vlm_options = copy.deepcopy(vlm_model_specs.SMOLDOCLING_TRANSFORMERS)
                logger.info("Using SmolDocling-Transformers (256M params, ~102s/page)")
            elif preset == "pixtral-12b-transformers" and hasattr(vlm_model_specs, 'PIXTRAL_12B_TRANSFORMERS'):
                vlm_options = copy.deepcopy(vlm_model_specs.PIXTRAL_12B_TRANSFORMERS)
                logger.info("Using Pixtral-12B-Transformers (12B params, ~1828s/page)")
            elif preset == "phi4" and hasattr(vlm_model_specs, 'PHI4_TRANSFORMERS'):
                vlm_options = copy.deepcopy(vlm_model_specs.PHI4_TRANSFORMERS)
                logger.info("Using Phi-4-Transformers (~1176s/page)")
            else:
                vlm_options = copy.deepcopy(vlm_model_specs.GRANITEDOCLING_TRANSFORMERS)
                logger.info("Using Granite-Docling-Transformers (MLX disabled)")

        elif self.hardware_config.device_type == HardwareType.CUDA:
            # CUDA GPU - use Transformers models with GPU acceleration
            if preset == "granite-vision" and hasattr(vlm_model_specs, 'GRANITE_VISION_TRANSFORMERS'):
                vlm_options = copy.deepcopy(vlm_model_specs.GRANITE_VISION_TRANSFORMERS)
                logger.info("Using Granite-Vision-Transformers with CUDA (2B params)")
            elif preset == "smoldocling" and hasattr(vlm_model_specs, 'SMOLDOCLING_TRANSFORMERS'):
                vlm_options = copy.deepcopy(vlm_model_specs.SMOLDOCLING_TRANSFORMERS)
                logger.info("Using SmolDocling-Transformers with CUDA (256M params)")
            elif preset == "pixtral-12b-transformers" and hasattr(vlm_model_specs, 'PIXTRAL_12B_TRANSFORMERS'):
                vlm_options = copy.deepcopy(vlm_model_specs.PIXTRAL_12B_TRANSFORMERS)
                logger.info("Using Pixtral-12B-Transformers with CUDA (12B params)")
            elif preset == "phi4" and hasattr(vlm_model_specs, 'PHI4_TRANSFORMERS'):
                vlm_options = copy.deepcopy(vlm_model_specs.PHI4_TRANSFORMERS)
                logger.info("Using Phi-4-Transformers with CUDA")
            else:
                vlm_options = copy.deepcopy(vlm_model_specs.GRANITEDOCLING_TRANSFORMERS)
                logger.info("Using Granite-Docling-Transformers with CUDA")

        else:
            # CPU fallback - use Transformers models
            if preset == "granite-vision" and hasattr(vlm_model_specs, 'GRANITE_VISION_TRANSFORMERS'):
                vlm_options = copy.deepcopy(vlm_model_specs.GRANITE_VISION_TRANSFORMERS)
                logger.info("Using Granite-Vision-Transformers on CPU (2B params)")
            elif preset == "smoldocling" and hasattr(vlm_model_specs, 'SMOLDOCLING_TRANSFORMERS'):
                vlm_options = copy.deepcopy(vlm_model_specs.SMOLDOCLING_TRANSFORMERS)
                logger.info("Using SmolDocling-Transformers on CPU (256M params)")
            else:
                vlm_options = copy.deepcopy(vlm_model_specs.GRANITEDOCLING_TRANSFORMERS)
                logger.info("Using Granite-Docling-Transformers on CPU")

        # Apply engineered prompt to improve extraction quality
        engineered_prompt = self._get_default_vlm_prompt()
        vlm_options.prompt = engineered_prompt

        # Configure generation parameters to reduce hallucination
        vlm_options.extra_generation_config = {
            "temperature": 0.1,  # Lower temperature for more deterministic output
            "do_sample": False,  # Disable sampling for greedy decoding
            "repetition_penalty": 1.2,  # Penalize repetition
        }

        logger.debug(f"Applied engineered prompt ({len(engineered_prompt)} chars)")
        logger.debug(f"Generation config: temp=0.1, do_sample=False, rep_penalty=1.2")

        # Create VlmPipelineOptions with selected model
        options = VlmPipelineOptions(
            vlm_options=vlm_options
        )

        logger.debug(f"VLM pipeline configured with model: {self.config.vlm.model_name}")
        return options

    def _get_ocr_options(self):
        """Get OCR options based on configured engine."""
        if self.config.ocr.engine == OcrEngine.EASYOCR:
            # Set model storage directory to cache location
            model_storage_dir = str(self.config.model_cache_dir / "artifacts" / "EasyOcr")
            Path(model_storage_dir).mkdir(parents=True, exist_ok=True)

            return EasyOcrOptions(
                lang=self.config.ocr.languages,
                use_gpu=self.config.ocr.use_gpu,
                download_enabled=True,  # Enable model downloads
                model_storage_directory=model_storage_dir,  # Explicit model storage path
            )
        elif self.config.ocr.engine == OcrEngine.TESSERACT:
            return TesseractOcrOptions(
                lang="+".join(self.config.ocr.languages),
            )
        elif self.config.ocr.engine == OcrEngine.OCRMAC:
            # OcrMac is macOS-specific
            try:
                from docling.datamodel.pipeline_options import OcrMacOptions

                return OcrMacOptions(
                    lang=self.config.ocr.languages,
                )
            except ImportError:
                logger.warning("OcrMac not available, falling back to EasyOCR")
                return EasyOcrOptions(
                    lang=self.config.ocr.languages,
                    use_gpu=False,
                )
        else:
            # Default fallback
            return EasyOcrOptions(
                lang=self.config.ocr.languages,
                use_gpu=self.config.ocr.use_gpu,
            )

    def _detect_document_type(self, source: Path) -> DocumentType:
        """
        Detect document type from file.

        Args:
            source: Path to document

        Returns:
            DocumentType
        """
        suffix = source.suffix.lower()

        # Map file extensions to document types
        type_map = {
            ".pdf": DocumentType.PDF_STANDARD,  # May be refined by content analysis
            ".docx": DocumentType.DOCX,
            ".pptx": DocumentType.PPTX,
            ".xlsx": DocumentType.XLSX,
            ".html": DocumentType.HTML,
            ".htm": DocumentType.HTML,
        }

        doc_type = type_map.get(suffix, DocumentType.UNKNOWN)

        # For PDFs, try to determine if scanned or has tables
        if doc_type == DocumentType.PDF_STANDARD:
            doc_type = self._analyze_pdf_content(source)

        return doc_type

    def _analyze_pdf_content(self, source: Path) -> DocumentType:
        """
        Analyze PDF content to determine type.

        Args:
            source: Path to PDF

        Returns:
            Refined DocumentType
        """
        # Basic heuristic: check file size and attempt to extract text
        try:
            import fitz  # PyMuPDF

            doc = fitz.open(source)

            if len(doc) == 0:
                return DocumentType.PDF_STANDARD

            # Sample first page
            first_page = doc[0]
            text = first_page.get_text()

            # Check if scanned (very little text)
            if len(text.strip()) < 50:
                logger.debug(f"PDF appears to be scanned (little text): {source.name}")
                return DocumentType.PDF_SCANNED

            # Check for tables (simple heuristic: look for table objects)
            tables = first_page.find_tables()
            # TableFinder object has tables attribute which is a list
            if tables and tables.tables and len(tables.tables) > 0:
                logger.debug(f"PDF contains tables: {source.name}")
                return DocumentType.PDF_TABLES

            doc.close()

        except ImportError:
            logger.warning("PyMuPDF not installed, using default PDF type")
        except Exception as e:
            logger.warning(f"Error analyzing PDF: {e}")

        return DocumentType.PDF_STANDARD

    def _get_default_vlm_prompt(self) -> str:
        """
        Get well-engineered default VLM prompt for document extraction.
        
        Based on Granite-Docling best practices and prompt engineering research:
        - Official Granite-Docling prompt format
        - Specific, structured instructions
        - Clear output format expectations
        
        Returns:
            Engineered prompt string optimized for document conversion
        """
        prompt = """Convert this document page to markdown format with the following requirements:

1. Extract ALL text content accurately, preserving the original structure and layout
2. Recognize and preserve document elements:
   - Headings and section titles (use appropriate markdown heading levels)
   - Paragraphs and text blocks (maintain spacing and organization)
   - Lists (bullet points and numbered lists)
   - Tables (convert to markdown table format)
   - Code blocks (use markdown code fences with language identifiers)
   - Mathematical formulas (convert to LaTeX notation)
   - Charts and figures (describe content and data)
3. Maintain document hierarchy and semantic structure
4. Preserve all information without omitting or summarizing content
5. Use proper markdown syntax for all elements
6. Do not add explanatory text, metadata, or comments about the conversion process
7. Output only the converted markdown content

Convert this page to markdown."""
        
        return prompt
