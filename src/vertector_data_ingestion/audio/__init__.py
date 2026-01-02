"""Audio processing module for transcription."""

from vertector_data_ingestion.audio.audio_factory import create_audio_transcriber
from vertector_data_ingestion.audio.base import (
    AudioTranscriber,
    TranscriptionResult,
    TranscriptionSegment,
)
from vertector_data_ingestion.audio.whisper_transcriber import WhisperTranscriber

__all__ = [
    "AudioTranscriber",
    "TranscriptionResult",
    "TranscriptionSegment",
    "WhisperTranscriber",
    "create_audio_transcriber",
]
