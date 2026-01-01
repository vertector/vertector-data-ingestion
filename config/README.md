# Configuration Directory

Configuration files and templates for Vertector Data Ingestion.

## Files

### .env.example

Template environment file with all available configuration options.

**Usage**:
```bash
# Copy to .env and customize
cp config/.env.example .env

# Edit with your settings
nano .env
```

**Note**: The `.env` file will be loaded automatically by Pydantic when you use `ConverterConfig()`.

## Configuration Methods

Vertector supports three configuration methods (in order of precedence):

1. **Programmatic** - Direct Python configuration
2. **Environment Variables** - Via `.env` file or shell exports
3. **Config Presets** - Pre-configured templates

## Environment-Specific Templates

### Apple Silicon (M1/M2/M3)

Create `.env` with:
```bash
VERTECTOR_VLM_USE_MLX=true
VERTECTOR_VLM_PRESET_MODEL=granite-mlx
VERTECTOR_AUDIO_BACKEND=mlx
VERTECTOR_OCR_ENGINE=ocrmac
VERTECTOR_BATCH_PROCESSING_WORKERS=4
```

### NVIDIA GPU Cloud

Create `.env` with:
```bash
VERTECTOR_VLM_USE_MLX=false
VERTECTOR_VLM_BATCH_SIZE=16
VERTECTOR_AUDIO_BACKEND=standard
VERTECTOR_OCR_ENGINE=easyocr
VERTECTOR_OCR_USE_GPU=true
VERTECTOR_BATCH_PROCESSING_WORKERS=8
```

### CPU-Only Cloud

Create `.env` with:
```bash
VERTECTOR_VLM_USE_MLX=false
VERTECTOR_VLM_BATCH_SIZE=4
VERTECTOR_AUDIO_BACKEND=standard
VERTECTOR_AUDIO_MODEL_SIZE=tiny
VERTECTOR_OCR_ENGINE=tesseract
VERTECTOR_OCR_USE_GPU=false
VERTECTOR_BATCH_PROCESSING_WORKERS=4
```

## See Also

- [Configuration Guide](../docs/configuration.md) - Complete configuration reference
- [QuickStart](../docs/quickstart.md) - Quick configuration examples
