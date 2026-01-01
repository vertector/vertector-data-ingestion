"""Test audio transcription with Whisper."""

import sys
from pathlib import Path
import urllib.request

sys.path.insert(0, str(Path(__file__).parent / "src"))

from vertector_data_ingestion.audio.whisper_transcriber import WhisperTranscriber

print("=" * 80)
print("AUDIO TRANSCRIPTION TEST - Whisper with MLX")
print("=" * 80)
print()

# Download Harvard sentences audio sample
test_dir = Path("test_documents")
test_dir.mkdir(exist_ok=True)
audio_file = test_dir / "harvard.wav"

if not audio_file.exists():
    print("Downloading Harvard sentences audio sample...")
    url = "https://storage.googleapis.com/kagglesdsdata/datasets/829978/1417968/harvard.wav?X-Goog-Algorithm=GOOG4-RSA-SHA256&X-Goog-Credential=databundle-worker-v2%40kaggle-161607.iam.gserviceaccount.com%2F20251229%2Fauto%2Fstorage%2Fgoog4_request&X-Goog-Date=20251229T225536Z&X-Goog-Expires=345600&X-Goog-SignedHeaders=host&X-Goog-Signature=abd8024b8667c0f39277c87fa904734bfac0eaf77dec6958dc7d433e4dfcc303cb274e6a8ee6df4be9b36f75d70f1250ac81ef6948c4eae78747baa3f1269721f0f502753bed4141dda1b38df34df21ed19e7ce95ee7d8a9a3c8f21bb5a87dec1f108fb20f8334302ef64acf2f7d718e266e8205fbb92333389ce0a997cc645aa29f3cfc65d94acf4249dc0d285ff74c6c15bebfe937429feaaa0b78e93de72887a4d0b90102e64d4df38374b21fb48545ab85ce7a0ae08b9b903020f9411dd9bcc5eb4313ae9e7de49850b7060292f5e0193c67e636cd77ebb275e3f5c477623f43e765c43204d905e842609755510264a0c96694d03642c473433886afa675"
    try:
        urllib.request.urlretrieve(url, audio_file)
        print(f"✅ Downloaded: {audio_file}")
        print(f"   Size: {audio_file.stat().st_size / 1024:.1f} KB")
    except Exception as e:
        print(f"❌ Error downloading audio: {e}")
        print("\nPlease provide an audio file at: test_documents/harvard.wav")
        print("Supported formats: WAV, MP3, M4A, FLAC, OGG")
        sys.exit(1)
else:
    print(f"Using existing audio file: {audio_file}")
    print(f"Size: {audio_file.stat().st_size / 1024:.1f} KB")

print()

# Test: MLX Whisper (base model)
print("=" * 80)
print("TEST: MLX Whisper - Base Model")
print("=" * 80)
print()

transcriber = WhisperTranscriber(
    model_name="base",
    use_mlx=True,
)

print(f"Backend: {transcriber.device}")
print(f"Model: {transcriber.model_name}")
print(f"Available: {transcriber.is_available()}")
print()

print("Transcribing audio file...")
try:
    result = transcriber.transcribe(audio_file)

    print()
    print("✅ Transcription Complete!")
    print(f"   Processing time: {result.duration:.2f}s")
    print(f"   Language detected: {result.language}")
    print(f"   Model used: {result.model_name}")
    print(f"   Number of segments: {len(result.segments)}")
    print(f"   Text length: {len(result.text)} characters")
    print()
    print("Full Transcription:")
    print("-" * 80)
    print(result.text)
    print("-" * 80)
    print()

    if result.segments:
        print("Timestamped Segments:")
        print("-" * 80)
        for i, seg in enumerate(result.segments, 1):
            timestamp = f"[{seg.start:.2f}s - {seg.end:.2f}s]"
            print(f"{i}. {timestamp:20s} {seg.text}")
        print("-" * 80)
        print()

    # Save transcription to file
    output_file = Path("audio_transcription_output.txt")
    with output_file.open('w') as f:
        f.write(f"Audio Transcription Results\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"File: {audio_file.name}\n")
        f.write(f"Model: {result.model_name}\n")
        f.write(f"Language: {result.language}\n")
        f.write(f"Processing time: {result.duration:.2f}s\n\n")
        f.write(f"Full Text:\n")
        f.write(f"{'-' * 80}\n")
        f.write(f"{result.text}\n")
        f.write(f"{'-' * 80}\n\n")

        if result.segments:
            f.write(f"Timestamped Segments:\n")
            f.write(f"{'-' * 80}\n")
            for i, seg in enumerate(result.segments, 1):
                f.write(f"{i}. [{seg.start:.2f}s - {seg.end:.2f}s] {seg.text}\n")

    print(f"✅ Saved detailed transcription to: {output_file}")

except Exception as e:
    print(f"❌ Error during transcription: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=" * 80)
print("AUDIO TRANSCRIPTION TEST COMPLETE")
print("=" * 80)
