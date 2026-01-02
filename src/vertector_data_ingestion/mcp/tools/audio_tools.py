"""Audio transcription tools for MCP server."""

import json
from pathlib import Path
from typing import Any, Dict, List
import asyncio

from vertector_data_ingestion import create_audio_transcriber
from vertector_data_ingestion.models.config import (
    AudioConfig,
    WhisperModelSize,
    AudioBackend,
)


async def transcribe_audio(
    file_path: str,
    model_size: str = "base",
    language: str = "auto",
    include_timestamps: bool = True,
    output_format: str = "text",
    backend: str = "auto",
) -> Dict[str, Any]:
    """Transcribe audio file to text.

    Args:
        file_path: Path to audio file
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code or 'auto' for detection
        include_timestamps: Include word/segment timestamps
        output_format: Output format (text, srt, json)
        backend: Whisper backend (auto, mlx, standard)

    Returns:
        Dict with transcription and metadata
    """
    try:
        path = Path(file_path)
        if not path.exists():
            return {
                "success": False,
                "error": f"File not found: {file_path}",
            }

        # Map model size
        model_size_map = {
            "tiny": WhisperModelSize.TINY,
            "base": WhisperModelSize.BASE,
            "small": WhisperModelSize.SMALL,
            "medium": WhisperModelSize.MEDIUM,
            "large": WhisperModelSize.LARGE,
        }
        whisper_model_size = model_size_map.get(model_size, WhisperModelSize.BASE)

        # Map backend
        backend_map = {
            "auto": AudioBackend.AUTO,
            "mlx": AudioBackend.MLX,
            "standard": AudioBackend.STANDARD,
        }
        whisper_backend = backend_map.get(backend, AudioBackend.AUTO)

        # Create audio config
        audio_config = AudioConfig(
            model_size=whisper_model_size,
            backend=whisper_backend,
            language=None if language == "auto" else language,
            word_timestamps=include_timestamps,
        )

        # Create transcriber
        transcriber = create_audio_transcriber(audio_config)

        # Transcribe
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            transcriber.transcribe,
            path,
        )

        # Format output based on requested format
        if output_format == "srt":
            # Format as SRT subtitles
            srt_content = []
            for idx, segment in enumerate(result.segments, 1):
                start_time = _format_timestamp(segment.start)
                end_time = _format_timestamp(segment.end)
                srt_content.append(f"{idx}\n{start_time} --> {end_time}\n{segment.text}\n")
            output = "\n".join(srt_content)

        elif output_format == "json":
            # Format as structured JSON
            output = {
                "text": result.text,
                "language": result.language,
                "duration": result.duration,
                "segments": [
                    {
                        "start": seg.start,
                        "end": seg.end,
                        "text": seg.text,
                    }
                    for seg in result.segments
                ],
            }

        else:  # text format
            output = result.text

        return {
            "success": True,
            "file_path": str(path),
            "transcription": output,
            "metadata": {
                "language": result.language,
                "duration": result.duration,
                "num_segments": len(result.segments),
                "model_size": model_size,
                "backend": backend,
            },
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "file_path": file_path,
        }


async def batch_transcribe_audio(
    file_paths: List[str],
    model_size: str = "base",
    language: str = "auto",
    include_timestamps: bool = True,
    output_format: str = "text",
    backend: str = "auto",
    max_workers: int = 2,
) -> Dict[str, Any]:
    """Transcribe multiple audio files in parallel.

    Args:
        file_paths: List of audio file paths
        model_size: Whisper model size (tiny, base, small, medium, large)
        language: Language code or 'auto' for detection
        include_timestamps: Include word/segment timestamps
        output_format: Output format (text, srt, json)
        backend: Whisper backend (auto, mlx, standard)
        max_workers: Number of parallel workers

    Returns:
        Dict with results for each audio file
    """
    try:
        # Transcribe all files in parallel (with limited concurrency)
        # Audio transcription is memory-intensive, so we limit workers
        tasks = [
            transcribe_audio(
                file_path=fp,
                model_size=model_size,
                language=language,
                include_timestamps=include_timestamps,
                output_format=output_format,
                backend=backend,
            )
            for fp in file_paths
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        successful = [r for r in results if isinstance(r, dict) and r.get("success")]
        failed = [r for r in results if not (isinstance(r, dict) and r.get("success"))]

        return {
            "success": True,
            "total": len(file_paths),
            "successful": len(successful),
            "failed": len(failed),
            "results": results,
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
        }


def _format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp (HH:MM:SS,mmm)."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
