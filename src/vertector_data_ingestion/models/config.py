"""Configuration models for Vertector Data Ingestion pipeline."""

from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class PipelineType(str, Enum):
    """Pipeline type enum."""

    CLASSIC = "classic"
    VLM = "vlm"


class OcrEngine(str, Enum):
    """OCR engine enum."""

    EASYOCR = "easyocr"
    TESSERACT = "tesseract"
    OCRMAC = "ocrmac"


class TableMode(str, Enum):
    """Table extraction mode."""

    FAST = "fast"
    ACCURATE = "accurate"


class VectorStoreType(str, Enum):
    """Vector store type."""

    CHROMA = "chroma"
    PINECONE = "pinecone"
    QDRANT = "qdrant"
    OPENSEARCH = "opensearch"


class WhisperModelSize(str, Enum):
    """Whisper model size options."""

    TINY = "tiny"
    BASE = "base"
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"
    TURBO = "turbo"


class AudioBackend(str, Enum):
    """Audio processing backend."""

    MLX = "mlx"  # Apple Silicon acceleration
    STANDARD = "standard"  # PyTorch (CUDA/CPU)
    AUTO = "auto"  # Auto-detect based on hardware


class ExportFormat(str, Enum):
    """Export format enum."""

    MARKDOWN = "markdown"
    JSON = "json"
    DOCTAGS = "doctags"


class OcrConfig(BaseSettings):
    """OCR configuration."""

    model_config = SettingsConfigDict(env_prefix="VERTECTOR_OCR_")

    engine: OcrEngine = OcrEngine.EASYOCR
    languages: List[str] = Field(default_factory=lambda: ["en"])
    use_gpu: bool = True
    confidence_threshold: float = 0.5


class TableConfig(BaseSettings):
    """Table extraction configuration."""

    model_config = SettingsConfigDict(env_prefix="VERTECTOR_TABLE_")

    mode: TableMode = TableMode.ACCURATE
    cell_matching: bool = Field(default=True, alias="do_cell_matching")


class AudioConfig(BaseSettings):
    """Audio transcription configuration."""

    model_config = SettingsConfigDict(env_prefix="VERTECTOR_AUDIO_")

    # Model selection
    model_size: WhisperModelSize = WhisperModelSize.BASE
    backend: AudioBackend = AudioBackend.AUTO

    # Transcription parameters
    language: Optional[str] = None  # Auto-detect if None (e.g., 'en', 'es', 'fr')
    word_timestamps: bool = True  # Enable word-level timestamps
    initial_prompt: Optional[str] = None  # Optional prompt to guide transcription

    # Performance settings
    beam_size: int = 5  # Beam search size (higher = more accurate, slower)
    temperature: float = 0.0  # Sampling temperature (0 = greedy, >0 = creative)

    # Advanced options
    vad_filter: bool = False  # Voice activity detection to filter silence
    condition_on_previous_text: bool = True  # Use previous text for context


class VlmConfig(BaseSettings):
    """VLM pipeline configuration."""

    model_config = SettingsConfigDict(env_prefix="VERTECTOR_VLM_")

    use_mlx: bool = False
    batch_size: int = 8
    model_name: str = "ibm-granite/granite-docling-258M"

    # Pre-configured model selection (alternative to custom_model_repo_id)
    # MLX models (Apple Silicon): "granite-mlx", "smoldocling-mlx", "qwen25-3b", "pixtral-12b", "gemma3-12b"
    # Transformers models (CUDA/CPU): "granite", "smoldocling", "granite-vision", "phi4", "pixtral-12b-transformers"
    preset_model: Optional[str] = None

    # Custom model configuration (optional - for non-default models)
    custom_model_repo_id: Optional[str] = None  # Hugging Face repo ID
    custom_model_prompt: Optional[str] = None  # Custom prompt for the model

    enable_picture_description: bool = True  # Enable AI-driven image captions
    enable_picture_classification: bool = True  # Enable image classification


class ChunkingConfig(BaseSettings):
    """Chunking configuration for RAG.

    The tokenizer model should be compatible with Hugging Face AutoTokenizer.
    Padding side is automatically detected based on the model name:
    - Qwen3 and Nemotron models use left padding
    - Other models use right padding (default)

    Recommended models:
        - Qwen/Qwen3-Embedding-0.6B (1024 dims, 32K context, left padding)
        - Qwen/Qwen3-Embedding-4B (2560 dims, 32K context, left padding)
        - Qwen/Qwen3-Embedding-8B (4096 dims, 32K context, left padding)
        - nvidia/llama-embed-nemotron-8b (4096 dims, left padding)
        - sentence-transformers/all-MiniLM-L6-v2 (384 dims, fast)
        - intfloat/multilingual-e5-large-instruct (1024 dims, multilingual)
    """

    model_config = SettingsConfigDict(env_prefix="VERTECTOR_CHUNK_")

    size: int = Field(default=2048, alias="chunk_size")
    tokenizer: str = Field(
        default="Qwen/Qwen3-Embedding-4B",
        description="Hugging Face tokenizer model (auto-detects padding side)"
    )
    merge_peers: bool = True
    max_tokens: int = 2048  # Default for compatibility, Qwen3-Embedding-4B supports 32,768


class VectorStoreConfig(BaseSettings):
    """Vector store configuration.

    The embedding model should be compatible with SentenceTransformers or
    Hugging Face AutoModel. The model will be used for encoding text into vectors.

    Recommended embedding models:
        - Qwen/Qwen3-Embedding-0.6B (1024 dims, 32K context, left padding)
        - Qwen/Qwen3-Embedding-4B (2560 dims, 32K context, left padding)
        - Qwen/Qwen3-Embedding-8B (4096 dims, 32K context, left padding)
        - nvidia/llama-embed-nemotron-8b (4096 dims, left padding)
        - sentence-transformers/all-MiniLM-L6-v2 (384 dims, fast)
        - intfloat/multilingual-e5-large-instruct (1024 dims, multilingual)
    """

    model_config = SettingsConfigDict(env_prefix="VERTECTOR_")

    vector_store: VectorStoreType = VectorStoreType.CHROMA
    embedding_model: str = Field(
        default="Qwen/Qwen3-Embedding-4B",
        description="Hugging Face embedding model for vector encoding"
    )
    chroma_persist_dir: Path = Field(default_factory=lambda: Path("./chroma_db"))


class ConverterConfig(BaseSettings):
    """Main converter configuration."""

    model_config = SettingsConfigDict(
        env_prefix="VERTECTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Pipeline selection
    default_pipeline: PipelineType = PipelineType.CLASSIC
    auto_detect_pipeline: bool = True

    # Model management
    auto_download_models: bool = True
    model_cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "vertector" / "models"
    )

    # Sub-configurations
    ocr: OcrConfig = Field(default_factory=OcrConfig)
    table: TableConfig = Field(default_factory=TableConfig)
    vlm: VlmConfig = Field(default_factory=VlmConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)

    # Error handling
    fail_fast: bool = False
    max_retries: int = 3

    # Caching
    enable_cache: bool = False
    cache_dir: Path = Field(
        default_factory=lambda: Path.home() / ".cache" / "vertector" / "conversions"
    )

    # Output directory
    output_dir: Path = Field(
        default_factory=lambda: Path("./output")
    )

    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Performance
    batch_processing_workers: int = 4


class LocalMpsConfig(ConverterConfig):
    """Configuration optimized for local macOS with MPS acceleration."""

    model_config = SettingsConfigDict(
        env_prefix="VERTECTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vlm: VlmConfig = Field(
        default_factory=lambda: VlmConfig(use_mlx=True, batch_size=8)
    )
    ocr: OcrConfig = Field(
        default_factory=lambda: OcrConfig(
            engine=OcrEngine.OCRMAC,
            use_gpu=True,
            languages=["en-US"]  # OCRMac requires locale-specific language codes
        )
    )
    audio: AudioConfig = Field(
        default_factory=lambda: AudioConfig(backend=AudioBackend.MLX)
    )


class CloudGpuConfig(ConverterConfig):
    """Configuration optimized for cloud deployment with GPU."""

    model_config = SettingsConfigDict(
        env_prefix="VERTECTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    vlm: VlmConfig = Field(
        default_factory=lambda: VlmConfig(use_mlx=False, batch_size=16)
    )
    ocr: OcrConfig = Field(
        default_factory=lambda: OcrConfig(engine=OcrEngine.EASYOCR, use_gpu=True)
    )
    auto_download_models: bool = False  # Require explicit model setup in production


class CloudCpuConfig(ConverterConfig):
    """Configuration optimized for cloud deployment without GPU."""

    model_config = SettingsConfigDict(
        env_prefix="VERTECTOR_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    default_pipeline: PipelineType = PipelineType.CLASSIC
    ocr: OcrConfig = Field(
        default_factory=lambda: OcrConfig(engine=OcrEngine.TESSERACT, use_gpu=False)
    )
    auto_download_models: bool = False  # Require explicit model setup in production
