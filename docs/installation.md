# Installation Guide

Complete installation instructions for Vertector Data Ingestion.

## Prerequisites

### Required

- **Python 3.10 or higher**
- **uv package manager** (recommended) or pip

### System Dependencies

#### macOS
```bash
# Install system dependencies
brew install cmake llvm

# For Tesseract OCR (optional)
brew install tesseract
```

#### Linux (Ubuntu/Debian)
```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install cmake llvm-dev build-essential

# For Tesseract OCR (optional)
sudo apt-get install tesseract-ocr
```

#### Windows
- Install CMake from https://cmake.org/download/
- Install LLVM from https://releases.llvm.org/
- For Tesseract OCR, download from https://github.com/UB-Mannheim/tesseract/wiki

### Optional (Based on Use Case)

- **Apple Silicon (M1/M2/M3/M4)**: For MLX acceleration
- **NVIDIA GPU**: For CUDA acceleration

## Quick Install

### Using uv (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd vertector-data-ingestion

# Install core dependencies only
uv sync

# Install with all optional features
uv sync --all-extras

# Install specific features
uv sync --extra asr         # Audio transcription
uv sync --extra mlx         # Apple Silicon acceleration
uv sync --extra notebooks   # Jupyter notebooks
uv sync --extra mcp         # MCP server support
uv sync --extra dev         # Development tools
```

### Using pip

```bash
# Clone the repository
git clone <repository-url>
cd vertector-data-ingestion

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core package
pip install -e .

# Install with all features
pip install -e ".[all]"

# Install specific features
pip install -e ".[asr]"        # Audio transcription
pip install -e ".[mlx]"        # Apple Silicon acceleration
pip install -e ".[notebooks]"  # Jupyter notebooks
pip install -e ".[mcp]"        # MCP server support
pip install -e ".[dev]"        # Development tools
```

## Optional Features

### Audio Transcription (`asr` extra)

Includes both MLX and standard Whisper backends:

```bash
# Using uv
uv sync --extra asr

# Using pip
pip install -e ".[asr]"
```

**Includes**:
- `openai-whisper` - Standard Whisper (CUDA/CPU)
- `mlx-whisper` - MLX-accelerated Whisper (Apple Silicon)

**Use when**:
- Running on NVIDIA GPU
- Running on CPU-only systems
- Need maximum compatibility

### OCR Engines

Install based on your preferred OCR engine:

#### EasyOCR (GPU-accelerated)

```bash
uv add easyocr
```

**Best for**:
- GPU-accelerated OCR
- Multi-language support
- High accuracy

#### Tesseract

```bash
# Install Tesseract system package first
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Then install Python bindings
uv add pytesseract
```

**Best for**:
- CPU-only environments
- Lightweight deployment
- Production stability

#### OCRMac (macOS only)

No additional installation needed - uses native macOS Vision framework.

**Best for**:
- macOS development
- Zero additional dependencies
- Native integration

### Vector Databases

Install your preferred vector store:

#### ChromaDB (Default)

```bash
uv add chromadb
```

#### Pinecone

```bash
uv add pinecone-client
```

#### Qdrant

```bash
uv add qdrant-client
```

#### OpenSearch

```bash
uv add opensearch-py
```

## Hardware-Specific Setup

### Apple Silicon (M1/M2/M3)

Optimal configuration for Apple Silicon:

```bash
# Install MLX-accelerated packages
uv add mlx-whisper

# Configure for MPS acceleration
export VERTECTOR_VLM_USE_MLX=true
export VERTECTOR_AUDIO_BACKEND=mlx
export VERTECTOR_OCR_ENGINE=ocrmac
```

Test your setup:

```python
from vertector_data_ingestion import HardwareDetector

info = HardwareDetector.get_device_info()
print(info)
# Expected: {'device_type': 'mps', 'has_mps': True, ...}
```

### NVIDIA GPU

Optimal configuration for CUDA:

```bash
# Install CUDA-enabled packages
uv add openai-whisper easyocr

# Verify CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Configure for GPU acceleration
export VERTECTOR_OCR_USE_GPU=true
export VERTECTOR_VLM_BATCH_SIZE=16
```

Test your setup:

```python
from vertector_data_ingestion import HardwareDetector

info = HardwareDetector.get_device_info()
print(info)
# Expected: {'device_type': 'cuda', 'has_cuda': True, ...}
```

### CPU-Only

Configuration for CPU-only environments:

```bash
# Install CPU-optimized packages
uv add openai-whisper pytesseract

# Install system dependencies
brew install tesseract  # macOS
# or
sudo apt-get install tesseract-ocr  # Ubuntu

# Configure for CPU
export VERTECTOR_AUDIO_BACKEND=standard
export VERTECTOR_OCR_ENGINE=tesseract
export VERTECTOR_OCR_USE_GPU=false
```

## Verification

### Basic Verification

```python
# Test basic imports
from vertector_data_ingestion import (
    UniversalConverter,
    HardwareDetector,
    LocalMpsConfig,
)

print("✅ Package installed successfully")
```

### Hardware Detection

```python
from vertector_data_ingestion import HardwareDetector

# Check detected hardware
config = HardwareDetector.detect()
print(f"Device: {config.device_type}")
print(f"MPS: {config.has_mps}")
print(f"CUDA: {config.has_cuda}")

# Get detailed info
info = HardwareDetector.get_device_info()
for key, value in info.items():
    print(f"{key}: {value}")
```

### Feature Testing

```python
from vertector_data_ingestion import UniversalConverter
from pathlib import Path

# Test document conversion
converter = UniversalConverter()
print("✅ Document processing ready")

# Test audio transcription (if installed)
try:
    from vertector_data_ingestion import create_audio_transcriber, AudioConfig

    transcriber = create_audio_transcriber(AudioConfig())
    if transcriber.is_available():
        print("✅ Audio transcription ready")
except ImportError as e:
    print(f"⚠️  Audio transcription not available: {e}")

# Test OCR (if needed)
try:
    from vertector_data_ingestion.ocr import create_ocr_engine, OcrConfig

    ocr = create_ocr_engine(OcrConfig())
    print("✅ OCR ready")
except ImportError as e:
    print(f"⚠️  OCR not available: {e}")
```

## Troubleshooting

### Import Errors

**Problem**: `ModuleNotFoundError: No module named 'vertector_data_ingestion'`

**Solution**:
```bash
# Reinstall in development mode
uv sync

# Or with pip
pip install -e .
```

### MLX Not Available

**Problem**: `ImportError: No module named 'mlx_whisper'`

**Solution**:
```bash
# Only works on Apple Silicon
uv add mlx-whisper

# Verify Apple Silicon
python -c "import platform; print(platform.processor())"
# Should show: arm
```

### CUDA Out of Memory

**Problem**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Reduce batch sizes in config
from vertector_data_ingestion import CloudGpuConfig

config = CloudGpuConfig()
config.vlm.batch_size = 4  # Reduce from default 16
config.audio.model_size = WhisperModelSize.BASE  # Use smaller model
```

### Tesseract Not Found

**Problem**: `TesseractNotFoundError`

**Solution**:
```bash
# Install system package
# macOS
brew install tesseract

# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# Verify installation
tesseract --version
```

### EasyOCR Language Pack Missing

**Problem**: EasyOCR fails to detect text

**Solution**:
```python
# EasyOCR downloads language packs on first use
# Ensure internet connection for first run

from vertector_data_ingestion.ocr import create_ocr_engine, OcrConfig

config = OcrConfig(
    engine="easyocr",
    languages=["en"]  # Specify required languages
)

ocr = create_ocr_engine(config)
# Language pack will be downloaded automatically
```

### Permission Errors

**Problem**: Permission denied when installing packages

**Solution**:
```bash
# Use uv (handles permissions automatically)
uv sync

# Or create virtual environment first
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Development Setup

For contributors and developers:

```bash
# Clone repository
git clone <repository-url>
cd vertector-data-ingestion

# Install with development dependencies
uv sync --extra dev

# Install pre-commit hooks (if available)
pre-commit install

# Run tests
uv run pytest

# Run tests with coverage
uv run pytest --cov=src/vertector_data_ingestion

# Run specific test
uv run pytest tests/integration/test_audio_transcription.py
```

## Docker Setup (Optional)

Coming soon: Docker images for containerized deployment.

## Next Steps

- Follow the [QuickStart Guide](quickstart.md) for your first conversion
- Read the [User Guide](user-guide.md) for detailed features
- Check [Configuration](configuration.md) for customization options

## Getting Help

If you encounter issues not covered here:

1. Check [Common Issues](#troubleshooting) above
2. Review [Known Limitations](user-guide.md#known-limitations)
3. Contact: cutetetteh@gmail.com
