"""Factory for creating audio transcribers from configuration."""

from loguru import logger

from vertector_data_ingestion.audio.base import AudioTranscriber
from vertector_data_ingestion.audio.whisper_transcriber import WhisperTranscriber
from vertector_data_ingestion.models.config import AudioConfig


def create_audio_transcriber(config: AudioConfig) -> AudioTranscriber:
    """
    Create an audio transcriber from configuration.

    Args:
        config: AudioConfig with transcription settings

    Returns:
        Configured AudioTranscriber instance

    Raises:
        ImportError: If required audio packages are not installed
    """
    logger.info(
        f"Creating audio transcriber: model={config.model_size.value}, "
        f"backend={config.backend.value}"
    )

    try:
        transcriber = WhisperTranscriber(config=config)

        if not transcriber.is_available():
            raise ImportError(
                f"Whisper backend '{config.backend.value}' is not available. "
                "Install required packages: pip install openai-whisper (standard) "
                "or pip install mlx-whisper (MLX for Apple Silicon)"
            )

        return transcriber

    except ImportError as e:
        logger.error(f"Failed to create audio transcriber: {e}")
        raise
