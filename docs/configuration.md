# Configuration Guide

Complete reference for configuring Vertector Data Ingestion.

## Table of Contents

- [Configuration Methods](#configuration-methods)
- [Configuration Presets](#configuration-presets)
- [Core Configuration](#core-configuration)
- [VLM Configuration](#vlm-configuration)
- [Audio Configuration](#audio-configuration)
- [OCR Configuration](#ocr-configuration)
- [Chunking Configuration](#chunking-configuration)
- [Environment Variables](#environment-variables)
- [Advanced Configuration](#advanced-configuration)

## Configuration Methods

Vertector supports three configuration methods (in order of precedence):

1. **Programmatic**: Direct Python configuration
2. **Environment Variables**: Via `.env` file or shell exports
3. **Config Presets**: Pre-configured templates

### 1. Programmatic Configuration

```python
from vertector_data_ingestion import ConverterConfig, UniversalConverter
from vertector_data_ingestion.models.config import (
    VlmConfig,
    AudioConfig,
    OcrConfig,
    WhisperModelSize,
)

config = ConverterConfig(
    vlm=VlmConfig(use_mlx=True, preset_model="granite-mlx"),
    audio=AudioConfig(model_size=WhisperModelSize.BASE),
    ocr=OcrConfig(engine="easyocr", use_gpu=True),
)

converter = UniversalConverter(config)
```

### 2. Environment Variables

```bash
# Create .env file
cat > .env << EOF
VERTECTOR_LOG_LEVEL=INFO
VERTECTOR_VLM_USE_MLX=true
VERTECTOR_AUDIO_MODEL_SIZE=base
VERTECTOR_OCR_ENGINE=easyocr
EOF

# Load automatically via Pydantic BaseSettings
```

```python
from vertector_data_ingestion import ConverterConfig, UniversalConverter

# Loads from environment variables automatically
config = ConverterConfig()
converter = UniversalConverter(config)
```

### 3. Config Presets

```python
from vertector_data_ingestion import LocalMpsConfig, UniversalConverter

# Use preset (can override specific settings)
config = LocalMpsConfig()
config.vlm.batch_size = 16  # Override preset value

converter = UniversalConverter(config)
```

## Configuration Presets

### LocalMpsConfig

Optimized for Apple Silicon development:

```python
from vertector_data_ingestion import LocalMpsConfig

config = LocalMpsConfig()
```

**Features**:
- MLX acceleration for VLM and audio
- OCRMac for native macOS OCR
- Optimized batch sizes for MPS
- Reduced workers for local machine

**Default Settings**:
```python
vlm:
  use_mlx: true
  preset_model: "granite-mlx"
  batch_size: 8

audio:
  backend: MLX
  model_size: BASE

ocr:
  engine: OCRMAC

batch_processing_workers: 4
```

### CloudGpuConfig

Optimized for CUDA/cloud GPU deployment:

```python
from vertector_data_ingestion import CloudGpuConfig

config = CloudGpuConfig()
```

**Features**:
- CUDA acceleration
- EasyOCR with GPU
- Larger batch sizes for GPU
- More parallel workers

**Default Settings**:
```python
vlm:
  use_mlx: false
  batch_size: 16

audio:
  backend: STANDARD  # Uses CUDA
  model_size: BASE

ocr:
  engine: EASYOCR
  use_gpu: true

batch_processing_workers: 8
```

### CloudCpuConfig

Optimized for CPU-only deployment:

```python
from vertector_data_ingestion import CloudCpuConfig

config = CloudCpuConfig()
```

**Features**:
- CPU-optimized settings
- Tesseract OCR
- Conservative batch sizes
- Reduced memory usage

**Default Settings**:
```python
vlm:
  use_mlx: false
  batch_size: 4

audio:
  backend: STANDARD  # Uses CPU
  model_size: BASE

ocr:
  engine: TESSERACT
  use_gpu: false

batch_processing_workers: 4
```

## Core Configuration

### ConverterConfig

Main configuration class:

```python
from pathlib import Path
from vertector_data_ingestion import ConverterConfig

config = ConverterConfig(
    # Logging
    log_level="INFO",  # DEBUG, INFO, WARNING, ERROR

    # Caching
    enable_cache=True,
    cache_dir=Path("./cache"),
    cache_ttl=86400,  # 24 hours in seconds

    # Performance
    batch_processing_workers=8,
    max_retries=3,
    retry_delay=1.0,

    # Nested configs
    vlm=VlmConfig(...),
    audio=AudioConfig(...),
    ocr=OcrConfig(...),
    chunking=ChunkingConfig(...),
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `log_level` | str | "INFO" | Logging level |
| `enable_cache` | bool | True | Enable result caching |
| `cache_dir` | Path | "./cache" | Cache directory |
| `cache_ttl` | int | 86400 | Cache time-to-live (seconds) |
| `batch_processing_workers` | int | 4 | Number of parallel workers |
| `max_retries` | int | 3 | Max retry attempts |
| `retry_delay` | float | 1.0 | Delay between retries (seconds) |

## VLM Configuration

### VlmConfig

Configure Vision-Language Model processing:

```python
from vertector_data_ingestion.models.config import VlmConfig

vlm_config = VlmConfig(
    # MLX acceleration (Apple Silicon)
    use_mlx=True,

    # Model selection
    preset_model="granite-mlx",  # or qwen25-3b, smoldocling, etc.

    # Custom model (alternative to preset)
    custom_model_repo_id="your-org/custom-model",
    custom_model_prompt="Custom prompt for your model",

    # Performance
    batch_size=8,

    # Features
    enable_picture_description=True,
    enable_table_detection=True,
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_mlx` | bool | False | Use MLX acceleration (Apple Silicon) |
| `preset_model` | str | None | Preset model name |
| `custom_model_repo_id` | str | None | HuggingFace repo ID for custom model |
| `custom_model_prompt` | str | None | Custom prompt for model |
| `batch_size` | int | 8 | Batch size for processing |
| `enable_picture_description` | bool | True | Enable image descriptions |
| `enable_table_detection` | bool | True | Enable table detection |

### Available Preset Models

| Preset Name | Model | Size | Best For |
|-------------|-------|------|----------|
| `granite-mlx` | IBM Granite Docling | 258M | General purpose (MLX) |
| `smoldocling` | SmolDocling | 248M | Fast processing |
| `qwen25-3b` | Qwen2.5-VL-3B | 3B | High accuracy |
| `pixtral` | Pixtral | Medium | Balanced |
| `gemma3` | Gemma 3 | Medium | Google ecosystem |
| `phi4` | Phi-4 | Medium | Microsoft ecosystem |

**Environment Variables**:
```bash
VERTECTOR_VLM_USE_MLX=true
VERTECTOR_VLM_PRESET_MODEL=granite-mlx
VERTECTOR_VLM_BATCH_SIZE=8
VERTECTOR_VLM_ENABLE_PICTURE_DESCRIPTION=true
```

## Audio Configuration

### AudioConfig

Configure audio transcription:

```python
from vertector_data_ingestion import AudioConfig, WhisperModelSize, AudioBackend

audio_config = AudioConfig(
    # Model selection
    model_size=WhisperModelSize.BASE,  # TINY, BASE, SMALL, MEDIUM, LARGE, TURBO

    # Backend selection
    backend=AudioBackend.AUTO,  # AUTO, MLX, STANDARD

    # Language
    language="en",  # None for auto-detect

    # Features
    word_timestamps=True,
    initial_prompt="Technical discussion",  # Context hint

    # Quality
    beam_size=5,  # 1-10, higher = more accurate, slower
    temperature=0.0,  # 0.0-1.0, 0.0 = deterministic

    # Advanced
    vad_filter=False,  # Voice activity detection
    condition_on_previous_text=True,  # Use context
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_size` | WhisperModelSize | BASE | Whisper model size |
| `backend` | AudioBackend | AUTO | Processing backend |
| `language` | str | None | Language code (None = auto-detect) |
| `word_timestamps` | bool | True | Enable word-level timestamps |
| `initial_prompt` | str | None | Context hint for better accuracy |
| `beam_size` | int | 5 | Beam search size (1-10) |
| `temperature` | float | 0.0 | Sampling temperature |
| `vad_filter` | bool | False | Filter silence |
| `condition_on_previous_text` | bool | True | Use previous text as context |

### Model Size Selection

| Model | Parameters | Speed | Accuracy | Memory |
|-------|-----------|-------|----------|---------|
| TINY | 39M | Fastest | Good | Low |
| BASE | 74M | Fast | Very Good | Low |
| SMALL | 244M | Medium | Excellent | Medium |
| MEDIUM | 769M | Slow | Excellent | High |
| LARGE | 1550M | Slowest | Best | Very High |
| TURBO | Optimized | Fast | Excellent | Medium |

**Environment Variables**:
```bash
VERTECTOR_AUDIO_MODEL_SIZE=base
VERTECTOR_AUDIO_BACKEND=mlx
VERTECTOR_AUDIO_LANGUAGE=en
VERTECTOR_AUDIO_WORD_TIMESTAMPS=true
VERTECTOR_AUDIO_BEAM_SIZE=5
VERTECTOR_AUDIO_TEMPERATURE=0.0
```

## OCR Configuration

### OcrConfig

Configure Optical Character Recognition:

```python
from vertector_data_ingestion.models.config import OcrConfig, OcrEngine

ocr_config = OcrConfig(
    # Engine selection
    engine=OcrEngine.EASYOCR,  # EASYOCR, TESSERACT, OCRMAC

    # GPU acceleration
    use_gpu=True,

    # Languages
    languages=["en", "es", "fr"],

    # Quality
    confidence_threshold=0.5,  # 0.0-1.0

    # Performance
    batch_size=4,
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `engine` | OcrEngine | EASYOCR | OCR engine to use |
| `use_gpu` | bool | False | Use GPU acceleration |
| `languages` | list[str] | ["en"] | Supported languages |
| `confidence_threshold` | float | 0.5 | Min confidence for text detection |
| `batch_size` | int | 4 | Batch size for processing |

### OCR Engine Comparison

| Engine | Speed | Accuracy | GPU | Platform |
|--------|-------|----------|-----|----------|
| EasyOCR | Medium | High | Yes | All |
| Tesseract | Fast | Good | No | All |
| OCRMac | Fast | Good | No | macOS only |

**Environment Variables**:
```bash
VERTECTOR_OCR_ENGINE=easyocr
VERTECTOR_OCR_USE_GPU=true
VERTECTOR_OCR_LANGUAGES=en,es,fr
VERTECTOR_OCR_CONFIDENCE_THRESHOLD=0.5
```

## Chunking Configuration

### ChunkingConfig

Configure RAG chunking strategy:

```python
from vertector_data_ingestion.models.config import ChunkingConfig

chunking_config = ChunkingConfig(
    # Size limits
    max_tokens=512,  # Maximum chunk size
    min_chunk_size=100,  # Minimum viable chunk

    # Overlap
    overlap_tokens=50,  # Overlap between chunks

    # Boundaries
    respect_boundaries=True,  # Respect section boundaries
    respect_sentences=True,  # Don't break mid-sentence

    # Metadata
    include_metadata=True,  # Include page, section metadata
)
```

**Parameters**:

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `max_tokens` | int | 512 | Maximum tokens per chunk |
| `min_chunk_size` | int | 100 | Minimum chunk size |
| `overlap_tokens` | int | 50 | Overlap between chunks |
| `respect_boundaries` | bool | True | Respect section boundaries |
| `respect_sentences` | bool | True | Don't break sentences |
| `include_metadata` | bool | True | Include metadata in chunks |

**Optimization Tips**:

- **More chunks** (smaller max_tokens): Better retrieval precision
- **Fewer chunks** (larger max_tokens): Faster embedding, more context
- **Higher overlap**: Better context continuity, more redundancy
- **Lower overlap**: Less redundancy, faster processing

**Environment Variables**:
```bash
VERTECTOR_CHUNKING_MAX_TOKENS=512
VERTECTOR_CHUNKING_MIN_CHUNK_SIZE=100
VERTECTOR_CHUNKING_OVERLAP_TOKENS=50
```

## Environment Variables

### Complete Reference

```bash
# General
VERTECTOR_LOG_LEVEL=INFO
VERTECTOR_ENABLE_CACHE=true
VERTECTOR_CACHE_DIR=./cache
VERTECTOR_CACHE_TTL=86400
VERTECTOR_BATCH_PROCESSING_WORKERS=8
VERTECTOR_MAX_RETRIES=3
VERTECTOR_RETRY_DELAY=1.0

# VLM
VERTECTOR_VLM_USE_MLX=true
VERTECTOR_VLM_PRESET_MODEL=granite-mlx
VERTECTOR_VLM_CUSTOM_MODEL_REPO_ID=
VERTECTOR_VLM_CUSTOM_MODEL_PROMPT=
VERTECTOR_VLM_BATCH_SIZE=8
VERTECTOR_VLM_ENABLE_PICTURE_DESCRIPTION=true
VERTECTOR_VLM_ENABLE_TABLE_DETECTION=true

# Audio
VERTECTOR_AUDIO_MODEL_SIZE=base
VERTECTOR_AUDIO_BACKEND=mlx
VERTECTOR_AUDIO_LANGUAGE=en
VERTECTOR_AUDIO_WORD_TIMESTAMPS=true
VERTECTOR_AUDIO_INITIAL_PROMPT=
VERTECTOR_AUDIO_BEAM_SIZE=5
VERTECTOR_AUDIO_TEMPERATURE=0.0
VERTECTOR_AUDIO_VAD_FILTER=false
VERTECTOR_AUDIO_CONDITION_ON_PREVIOUS_TEXT=true

# OCR
VERTECTOR_OCR_ENGINE=easyocr
VERTECTOR_OCR_USE_GPU=true
VERTECTOR_OCR_LANGUAGES=en
VERTECTOR_OCR_CONFIDENCE_THRESHOLD=0.5
VERTECTOR_OCR_BATCH_SIZE=4

# Chunking
VERTECTOR_CHUNKING_MAX_TOKENS=512
VERTECTOR_CHUNKING_MIN_CHUNK_SIZE=100
VERTECTOR_CHUNKING_OVERLAP_TOKENS=50
VERTECTOR_CHUNKING_RESPECT_BOUNDARIES=true
VERTECTOR_CHUNKING_RESPECT_SENTENCES=true
VERTECTOR_CHUNKING_INCLUDE_METADATA=true

# Vector Store
VERTECTOR_VECTOR_STORE=chroma
VERTECTOR_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-0.6B
VERTECTOR_CHROMA_PERSIST_DIRECTORY=./chroma_db
VERTECTOR_CHROMA_COLLECTION_NAME=documents
```

## Advanced Configuration

### Custom VLM Models

Use your own fine-tuned models:

```python
from vertector_data_ingestion.models.config import VlmConfig

vlm_config = VlmConfig(
    custom_model_repo_id="your-org/fine-tuned-docling",
    custom_model_prompt="<custom>Your specialized prompt</custom>",
    batch_size=16,
)
```

### Hardware-Specific Tuning

#### Apple Silicon (M1/M2/M3)

```python
config = LocalMpsConfig()
config.vlm.batch_size = 16  # Increase for M3 Max
config.audio.model_size = WhisperModelSize.SMALL  # Larger model for better accuracy
```

#### NVIDIA GPU (16GB+ VRAM)

```python
config = CloudGpuConfig()
config.vlm.batch_size = 32  # Increase for high VRAM
config.audio.model_size = WhisperModelSize.MEDIUM
```

#### CPU-Only (Memory Constrained)

```python
config = CloudCpuConfig()
config.vlm.batch_size = 2  # Reduce for low memory
config.audio.model_size = WhisperModelSize.TINY
config.batch_processing_workers = 2
```

### Multi-Language Setup

```python
from vertector_data_ingestion import ConverterConfig
from vertector_data_ingestion.models.config import OcrConfig, AudioConfig

config = ConverterConfig(
    ocr=OcrConfig(
        engine="easyocr",
        languages=["en", "es", "fr", "de", "zh"],
        use_gpu=True,
    ),
    audio=AudioConfig(
        language=None,  # Auto-detect
        word_timestamps=True,
    ),
)
```

### Production Deployment

```python
from pathlib import Path
from vertector_data_ingestion import CloudGpuConfig

config = CloudGpuConfig()

# Increase workers for high throughput
config.batch_processing_workers = 16

# Enable caching
config.enable_cache = True
config.cache_dir = Path("/mnt/cache")
config.cache_ttl = 604800  # 7 days

# Optimize for quality
config.vlm.batch_size = 32
config.audio.model_size = WhisperModelSize.SMALL
config.audio.beam_size = 10

# Increase retries
config.max_retries = 5
config.retry_delay = 2.0
```

## Next Steps

- See [User Guide](user-guide.md) for feature usage
- Check [API Reference](api-reference.md) for detailed API docs
- Review [examples/](../examples/) for configuration examples

## Getting Help

Contact: cutetetteh@gmail.com
