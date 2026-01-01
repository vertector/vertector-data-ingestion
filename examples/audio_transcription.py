"""Audio transcription example with Whisper."""

from pathlib import Path

from vertector_data_ingestion import (
    create_audio_transcriber,
    AudioConfig,
    WhisperModelSize,
    AudioBackend,
    setup_logging,
)


def main():
    """Demonstrate audio transcription features."""

    setup_logging(log_level="INFO")

    # Example 1: Basic transcription
    print("=" * 60)
    print("Example 1: Basic Audio Transcription")
    print("=" * 60)

    config = AudioConfig(
        model_size=WhisperModelSize.BASE,
        backend=AudioBackend.AUTO,  # Auto-detect MLX/CUDA/CPU
    )

    transcriber = create_audio_transcriber(config)

    audio_file = Path("path/to/your/audio.wav")

    if audio_file.exists():
        result = transcriber.transcribe(audio_file)

        print(f"\nTranscription:")
        print(result.text)
        print(f"\nLanguage: {result.language}")
        print(f"Duration: {result.duration:.2f}s")
        print(f"Model: {result.model_name}")

        # Example 2: Access timestamped segments
        print("\n" + "=" * 60)
        print("Example 2: Timestamped Segments")
        print("=" * 60)

        for i, segment in enumerate(result.segments[:5], 1):
            print(f"\nSegment {i}:")
            print(f"  Time: {segment.start:.1f}s - {segment.end:.1f}s")
            print(f"  Text: {segment.text}")

    else:
        print(f"Audio file not found: {audio_file}")
        demonstrate_without_file()


def demonstrate_without_file():
    """Demonstrate configuration options without actual file."""

    print("\n" + "=" * 60)
    print("Configuration Examples")
    print("=" * 60)

    # High accuracy configuration
    print("\n1. High Accuracy Config:")
    high_accuracy = AudioConfig(
        model_size=WhisperModelSize.LARGE,
        language="en",
        beam_size=10,  # Higher beam size for better accuracy
        temperature=0.0,  # Deterministic output
        word_timestamps=True,
    )
    print(f"   Model: {high_accuracy.model_size.value}")
    print(f"   Beam size: {high_accuracy.beam_size}")

    # Fast transcription configuration
    print("\n2. Fast Transcription Config:")
    fast_config = AudioConfig(
        model_size=WhisperModelSize.TINY,
        backend=AudioBackend.AUTO,
        word_timestamps=False,  # Faster without timestamps
    )
    print(f"   Model: {fast_config.model_size.value}")
    print(f"   Backend: {fast_config.backend.value}")

    # Multi-language configuration
    print("\n3. Multi-Language Config:")
    multilang_config = AudioConfig(
        model_size=WhisperModelSize.SMALL,
        language=None,  # Auto-detect language
        word_timestamps=True,
    )
    print(f"   Language: Auto-detect")
    print(f"   Model: {multilang_config.model_size.value}")

    # Technical content configuration
    print("\n4. Technical Content Config:")
    technical_config = AudioConfig(
        model_size=WhisperModelSize.BASE,
        initial_prompt="Technical discussion about machine learning and AI",
        language="en",
        word_timestamps=True,
        beam_size=8,
    )
    print(f"   Context prompt: {technical_config.initial_prompt}")
    print(f"   Beam size: {technical_config.beam_size}")


def batch_transcription_example():
    """Example of transcribing multiple audio files."""

    print("\n" + "=" * 60)
    print("Example 3: Batch Transcription")
    print("=" * 60)

    config = AudioConfig(model_size=WhisperModelSize.BASE)
    transcriber = create_audio_transcriber(config)

    audio_files = [
        Path("audio1.wav"),
        Path("audio2.wav"),
        Path("audio3.wav"),
    ]

    results = []
    for audio_file in audio_files:
        if audio_file.exists():
            result = transcriber.transcribe(audio_file)
            results.append((audio_file.name, result))
            print(f"\n{audio_file.name}:")
            print(f"  Length: {result.duration:.1f}s")
            print(f"  Text preview: {result.text[:100]}...")

    return results


def save_transcription_example():
    """Example of saving transcription to file."""

    print("\n" + "=" * 60)
    print("Example 4: Save Transcription")
    print("=" * 60)

    config = AudioConfig(model_size=WhisperModelSize.BASE)
    transcriber = create_audio_transcriber(config)

    audio_file = Path("path/to/audio.wav")
    output_dir = Path("transcriptions")

    if audio_file.exists():
        result = transcriber.transcribe(audio_file)

        # Create output directory
        output_dir.mkdir(exist_ok=True)

        # Save full transcription
        text_file = output_dir / f"{audio_file.stem}.txt"
        with open(text_file, "w") as f:
            f.write(result.text)

        # Save with timestamps
        srt_file = output_dir / f"{audio_file.stem}.srt"
        with open(srt_file, "w") as f:
            for i, segment in enumerate(result.segments, 1):
                start = format_timestamp(segment.start)
                end = format_timestamp(segment.end)
                f.write(f"{i}\n")
                f.write(f"{start} --> {end}\n")
                f.write(f"{segment.text}\n\n")

        print(f"Saved transcription to {text_file}")
        print(f"Saved SRT subtitles to {srt_file}")


def format_timestamp(seconds: float) -> str:
    """Format seconds as SRT timestamp."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


if __name__ == "__main__":
    main()
    batch_transcription_example()
    save_transcription_example()
