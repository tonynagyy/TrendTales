import sys
import re
from pathlib import Path
# pyrefly: ignore [missing-import]
from gtts import gTTS

# Force UTF-8 encoding on standard output/error to prevent Windows charmap errors
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

OUTPUT_DIR = Path("outputs")
DEFAULT_OUTPUT = OUTPUT_DIR / "generated_story.mp3"


def _sanitize_text(text: str) -> str:
    """Remove characters that Windows codecs / gTTS can't encode."""
    # Replace common unicode symbols with ASCII equivalents
    replacements = {
        "\u2713": "", "\u2714": "", "\u2715": "", "\u2716": "",  # ✓ ✔ ✕ ✖
        "\u2019": "'", "\u2018": "'", "\u201c": '"', "\u201d": '"',  # curly quotes
        "\u2014": "-", "\u2013": "-",  # em/en dash
        "\u2026": "...",  # ellipsis
    }
    for char, replacement in replacements.items():
        text = text.replace(char, replacement)
    # Strip any remaining non-ASCII characters
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()


def text_to_speech(
    text: str,
    output_path: str = str(DEFAULT_OUTPUT),
    lang: str = "en",
    slow: bool = False,
) -> str:
    """
    Convert generated story text to an MP3 audio file using gTTS.
    """

    if not text or not text.strip():
        raise ValueError("Story text cannot be empty.")

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("STEP - TEXT TO SPEECH")
    print("=" * 60)

    print(f"Generating audio...")
    print(f"Output: {output_file}")

    clean_text = _sanitize_text(text)
    if not clean_text:
        raise ValueError("Story text is empty after sanitization.")

    tts = gTTS(
        text=clean_text,
        lang=lang,
        slow=slow,
    )

    tts.save(str(output_file))

    print("Audio generated successfully.")
    print(f"Saved to: {output_file.resolve()}")

    return str(output_file)


if __name__ == "__main__":
    test_text = (
        "Welcome to TrendTales. "
        "This is a test of the text to speech system."
    )

    audio_path = text_to_speech(test_text)

    print(f"\nTest completed: {audio_path}")
