"""Base audio transcription interface."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TranscriptionSegment:
    """A segment of transcribed audio with timestamps."""

    start: float
    end: float
    text: str


@dataclass
class TranscriptionResult:
    """Result of audio transcription."""

    text: str
    language: str
    segments: list[TranscriptionSegment]
    duration: float
    model_name: str


class AudioTranscriber(ABC):
    """Base class for audio transcription."""

    @abstractmethod
    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        """
        Transcribe audio file to text.

        Args:
            audio_path: Path to audio file
            language: Optional language code (e.g., 'en', 'es')

        Returns:
            TranscriptionResult with text, language, segments, and metadata
        """
        raise NotImplementedError

    @abstractmethod
    def is_available(self) -> bool:
        """Check if transcriber is available."""
        raise NotImplementedError
