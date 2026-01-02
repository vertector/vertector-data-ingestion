"""Whisper-based audio transcription with MLX support for Apple Silicon."""

import time
from pathlib import Path

from loguru import logger

from vertector_data_ingestion.audio.base import (
    AudioTranscriber,
    TranscriptionResult,
    TranscriptionSegment,
)
from vertector_data_ingestion.core.hardware_detector import HardwareDetector, HardwareType
from vertector_data_ingestion.models.config import AudioBackend, AudioConfig


class WhisperTranscriber(AudioTranscriber):
    """Whisper-based audio transcriber with hardware acceleration."""

    def __init__(
        self,
        config: AudioConfig | None = None,
        model_name: str | None = None,
        use_mlx: bool | None = None,
        device: str | None = None,
    ):
        """
        Initialize Whisper transcriber.

        Args:
            config: AudioConfig object (preferred method)
            model_name: Whisper model size (legacy, use config instead)
            use_mlx: Use MLX acceleration (legacy, use config instead)
            device: Device to use (legacy, use config instead)
        """
        # Use config if provided, otherwise fall back to legacy parameters
        if config is None:
            config = AudioConfig()
            if model_name is not None:
                # Handle both string and enum
                from vertector_data_ingestion.models.config import WhisperModelSize

                if isinstance(model_name, str):
                    config.model_size = WhisperModelSize(model_name)
                else:
                    config.model_size = model_name
            if use_mlx is not None:
                config.backend = AudioBackend.MLX if use_mlx else AudioBackend.STANDARD

        self.config = config
        self.model_name = config.model_size.value
        self.hardware_config = HardwareDetector.detect()
        self.device = device or self._get_device()
        self.model = None

        logger.info(
            f"Initializing WhisperTranscriber with model={self.model_name}, "
            f"device={self.device}, backend={config.backend.value}"
        )

    def _get_device(self) -> str:
        """Determine the best device for transcription."""
        # Handle AUTO backend - auto-detect best device
        if self.config.backend == AudioBackend.AUTO:
            if self.hardware_config.device_type == HardwareType.MPS:
                return "mlx"
            elif self.hardware_config.device_type == HardwareType.CUDA:
                return "cuda"
            else:
                return "cpu"

        # Handle explicit backend selection
        if self.config.backend == AudioBackend.MLX:
            if self.hardware_config.device_type == HardwareType.MPS:
                return "mlx"
            else:
                logger.warning("MLX backend requested but MPS not available, falling back to CPU")
                return "cpu"

        # STANDARD backend
        if self.hardware_config.device_type == HardwareType.CUDA:
            return "cuda"
        return "cpu"

    def _load_model(self):
        """Load the Whisper model with appropriate backend."""
        if self.model is not None:
            return

        logger.info(f"Loading Whisper model: {self.model_name}")

        if self.device == "mlx" and self.hardware_config.device_type == HardwareType.MPS:
            # Use MLX Whisper for Apple Silicon
            try:
                import mlx_whisper

                self.model = mlx_whisper
                self.backend = "mlx"
                logger.info(f"Loaded MLX Whisper model: {self.model_name}")
            except ImportError as e:
                logger.warning(f"MLX Whisper not available: {e}, falling back to standard Whisper")
                self.backend = "standard"
                import whisper

                self.model = whisper.load_model(self.model_name, device="cpu")
        else:
            # Use standard Whisper
            import whisper

            device_map = {
                "mps": "cpu",  # Standard whisper doesn't support MPS well
                "cuda": "cuda",
                "cpu": "cpu",
            }
            actual_device = device_map.get(self.device, "cpu")
            self.model = whisper.load_model(self.model_name, device=actual_device)
            self.backend = "standard"
            logger.info(f"Loaded standard Whisper model on {actual_device}")

    def transcribe(self, audio_path: Path, language: str | None = None) -> TranscriptionResult:
        """
        Transcribe audio file using Whisper.

        Args:
            audio_path: Path to audio file (mp3, wav, m4a, etc.)
            language: Optional language code (overrides config if provided)

        Returns:
            TranscriptionResult with full transcription and segments
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_model()

        # Use language from parameter or config
        transcribe_language = language or self.config.language

        logger.info(f"Transcribing {audio_path.name} with {self.backend} backend")
        start_time = time.time()

        if self.backend == "mlx":
            # MLX Whisper transcription
            # MLX models use format: mlx-community/whisper-{model_name}-mlx
            mlx_repo = f"mlx-community/whisper-{self.model_name}-mlx"
            result = self.model.transcribe(
                str(audio_path),
                path_or_hf_repo=mlx_repo,
                language=transcribe_language,
                word_timestamps=self.config.word_timestamps,
            )
        else:
            # Standard Whisper transcription
            transcribe_kwargs = {
                "language": transcribe_language,
                "word_timestamps": self.config.word_timestamps,
                "verbose": False,
                "beam_size": self.config.beam_size,
                "temperature": self.config.temperature,
                "condition_on_previous_text": self.config.condition_on_previous_text,
            }

            if self.config.initial_prompt:
                transcribe_kwargs["initial_prompt"] = self.config.initial_prompt

            if self.config.vad_filter:
                transcribe_kwargs["vad_filter"] = True

            result = self.model.transcribe(str(audio_path), **transcribe_kwargs)

        duration = time.time() - start_time

        # Extract segments with timestamps
        segments = []
        if "segments" in result:
            for seg in result["segments"]:
                segments.append(
                    TranscriptionSegment(
                        start=seg.get("start", 0.0),
                        end=seg.get("end", 0.0),
                        text=seg.get("text", "").strip(),
                    )
                )

        transcription_result = TranscriptionResult(
            text=result["text"].strip(),
            language=result.get("language", language or "unknown"),
            segments=segments,
            duration=duration,
            model_name=f"whisper-{self.model_name}-{self.backend}",
        )

        logger.info(
            f"Transcription complete in {duration:.2f}s: "
            f"{len(transcription_result.text)} chars, "
            f"{len(segments)} segments"
        )

        return transcription_result

    def is_available(self) -> bool:
        """Check if Whisper is available."""
        try:
            if self.device == "mlx":
                import mlx_whisper

                return True
            else:
                import whisper

                return True
        except ImportError:
            return False
