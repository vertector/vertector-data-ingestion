"""Base audio transcription interface."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass


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
    segments: List[TranscriptionSegment]
    duration: float
    model_name: str


class AudioTranscriber(ABC):
    """Base class for audio transcription."""

    @abstractmethod
    def transcribe(self, audio_path: Path, language: Optional[str] = None) -> TranscriptionResult:
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
