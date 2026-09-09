from pathlib import Path
from gtts import gTTS


OUTPUT_DIR = Path("outputs")
DEFAULT_OUTPUT = OUTPUT_DIR / "generated_story.mp3"


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

    tts = gTTS(
        text=text.strip(),
        lang=lang,
        slow=slow,
    )

    tts.save(str(output_file))

    print("✓ Audio generated successfully.")
    print(f"✓ Saved to: {output_file.resolve()}")

    return str(output_file)


if __name__ == "__main__":
    test_text = (
        "Welcome to TrendTales. "
        "This is a test of the text to speech system."
    )

    audio_path = text_to_speech(test_text)

    print(f"\nTest completed: {audio_path}")
