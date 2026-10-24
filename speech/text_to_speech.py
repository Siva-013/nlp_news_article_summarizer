"""
Text-to-Speech (TTS) Integration Module
Part of Phase 10: Text-to-Speech (TTS) Integration

This module converts extractive news summaries into spoken audio using classical,
offline speech synthesis via `pyttsx3` (Windows SAPI5 / POSIX NSSpeechSynthesizer / espeak).

Key Architectural Guarantees:
1. Strict Phase Isolation: Standalone speech synthesis without LLMs, transformers,
   or external cloud AI speech APIs.
2. Offline & Private: No internet connectivity required.
3. Deterministic Validation: Strict validation of speech rates (50-400 wpm),
   volumes (0.0-1.0), and available voice identifiers.
4. Robust File Output: Generates non-empty audio files (WAV) with verified file integrity.
5. Testability: Supports dependency injection of mock engines for headless test environments.
"""

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Union
import threading
import platform

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


# Default configuration parameters
DEFAULT_RATE = 175          # Words per minute (standard conversational pace)
DEFAULT_VOLUME = 1.0        # Maximum volume (0.0 to 1.0)
MIN_RATE = 50
MAX_RATE = 400
SUPPORTED_EXTENSIONS = {".wav"}


class TextToSpeechEngine:
    """
    Offline Text-to-Speech synthesis engine wrapping pyttsx3.
    """

    def __init__(
        self,
        rate: int = DEFAULT_RATE,
        volume: float = DEFAULT_VOLUME,
        voice_id: Optional[str] = None,
        engine: Optional[Any] = None,
        auto_init: bool = True,
    ):
        """
        Initialize the TextToSpeechEngine.

        Args:
            rate: Speech rate in words per minute (range: 50-400).
            volume: Speech volume (range: 0.0-1.0).
            voice_id: Specific voice identifier string (optional).
            engine: Optional pre-configured or mocked speech engine instance.
            auto_init: Whether to initialize the underlying engine immediately.
        """
        self._engine = engine
        self._rate = self.validate_rate(rate)
        self._volume = self.validate_volume(volume)
        self._voice_id = voice_id

        if auto_init and self._engine is None:
            self._init_engine()

    def _init_engine(self) -> Any:
        """Initialize the pyttsx3 speech engine instance if not already initialized."""
        if self._engine is not None:
            return self._engine

        if pyttsx3 is None:
            raise ImportError(
                "The 'pyttsx3' package is required for Text-to-Speech synthesis.\n"
                "Please install it using: pip install pyttsx3"
            )

        try:
            self._engine = pyttsx3.init()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to initialize system speech engine: {exc}"
            ) from exc

        # Apply initial configuration
        self._apply_properties()
        return self._engine

    def _apply_properties(self) -> None:
        """Apply current rate, volume, and voice settings to the engine."""
        if self._engine is None:
            return

        try:
            self._engine.setProperty("rate", self._rate)
            self._engine.setProperty("volume", self._volume)
            if self._voice_id:
                self.set_voice(self._voice_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to configure speech engine properties: {exc}") from exc

    # =========================================================================
    # VALIDATION METHODS
    # =========================================================================

    @staticmethod
    def validate_text(text: Any) -> str:
        """
        Validate input summary text.

        Args:
            text: Input text to validate.

        Returns:
            Normalized string without leading/trailing whitespace.

        Raises:
            TypeError: If input is not a string.
            ValueError: If input is empty or contains only whitespace.
        """
        if not isinstance(text, str):
            raise TypeError(
                f"Text input must be a string, got {type(text).__name__}."
            )

        cleaned = text.strip()
        if not cleaned:
            raise ValueError("Text input cannot be empty or contain only whitespace.")

        return cleaned

    @staticmethod
    def validate_rate(rate: int) -> int:
        """
        Validate speech rate in words per minute.

        Args:
            rate: Integer rate.

        Returns:
            Validated rate integer.

        Raises:
            TypeError: If rate is not an integer.
            ValueError: If rate is out of bounds (50-400 wpm).
        """
        if not isinstance(rate, int) or isinstance(rate, bool):
            raise TypeError(f"Speech rate must be an integer, got {type(rate).__name__}.")

        if rate < MIN_RATE or rate > MAX_RATE:
            raise ValueError(
                f"Speech rate {rate} is out of bounds. Supported range is [{MIN_RATE}, {MAX_RATE}] wpm."
            )
        return rate

    @staticmethod
    def validate_volume(volume: Union[int, float]) -> float:
        """
        Validate speech volume.

        Args:
            volume: Float between 0.0 and 1.0.

        Returns:
            Validated float volume.

        Raises:
            TypeError: If volume is not a float or integer.
            ValueError: If volume is out of bounds [0.0, 1.0].
        """
        if not isinstance(volume, (int, float)) or isinstance(volume, bool):
            raise TypeError(
                f"Volume must be a numeric value, got {type(volume).__name__}."
            )

        vol_float = float(volume)
        if vol_float < 0.0 or vol_float > 1.0:
            raise ValueError(
                f"Volume {vol_float} is out of bounds. Supported range is [0.0, 1.0]."
            )
        return vol_float

    # =========================================================================
    # CONFIGURATION API
    # =========================================================================

    @property
    def rate(self) -> int:
        """Current speech rate in words per minute."""
        return self._rate

    @property
    def volume(self) -> float:
        """Current speech volume (0.0 to 1.0)."""
        return self._volume

    @property
    def voice_id(self) -> Optional[str]:
        """Currently configured voice identifier."""
        return self._voice_id

    def set_rate(self, rate: int) -> None:
        """
        Configure speech rate.

        Args:
            rate: Rate in words per minute (50-400).
        """
        self._rate = self.validate_rate(rate)
        if self._engine is not None:
            self._engine.setProperty("rate", self._rate)

    def set_volume(self, volume: float) -> None:
        """
        Configure speech volume.

        Args:
            volume: Float between 0.0 and 1.0.
        """
        self._volume = self.validate_volume(volume)
        if self._engine is not None:
            self._engine.setProperty("volume", self._volume)

    def set_voice(self, voice_id: str) -> None:
        """
        Select a voice by identifier. Validates that the requested voice exists.

        Args:
            voice_id: String identifier of the target voice.

        Raises:
            ValueError: If the requested voice ID is not available on the system.
        """
        if not voice_id or not isinstance(voice_id, str):
            raise ValueError("Voice ID must be a non-empty string.")

        available_voices = self.get_available_voices()
        available_ids = {v["id"] for v in available_voices}

        if available_ids and voice_id not in available_ids:
            raise ValueError(
                f"Requested voice ID '{voice_id}' is not available on this system. "
                f"Available voices: {list(available_ids)}"
            )

        self._voice_id = voice_id
        if self._engine is not None:
            self._engine.setProperty("voice", self._voice_id)

    def get_available_voices(self) -> List[Dict[str, Any]]:
        """
        Query and return all available speech synthesis voices on the current system.

        Returns:
            List of dictionaries containing 'id', 'name', 'languages', and 'gender'.
        """
        engine = self._init_engine()
        try:
            voices = engine.getProperty("voices")
        except Exception as exc:
            raise RuntimeError(f"Failed to query system voices: {exc}") from exc

        voice_list: List[Dict[str, Any]] = []
        if not voices:
            return voice_list

        for v in voices:
            v_id = getattr(v, "id", str(v))
            v_name = getattr(v, "name", "Unknown Voice")
            v_langs = getattr(v, "languages", [])
            v_gender = getattr(v, "gender", None)

            voice_list.append(
                {
                    "id": v_id,
                    "name": v_name,
                    "languages": v_langs if isinstance(v_langs, list) else [str(v_langs)],
                    "gender": v_gender,
                }
            )

        return voice_list

    # =========================================================================
    # SYNTHESIS OPERATIONS
    # =========================================================================

    def speak(self, text: str) -> None:
        """
        Synthesize text and play spoken audio aloud through system speakers.

        Args:
            text: Summary text to speak.

        Raises:
            ValueError / TypeError: On invalid text input.
            RuntimeError: On engine speech synthesis failure.
        """
        validated_text = self.validate_text(text)
        engine = self._init_engine()

        try:
            engine.say(validated_text)
            engine.runAndWait()
        except Exception as exc:
            raise RuntimeError(f"Speech playback failed: {exc}") from exc

    def save_to_file(
        self,
        text: str,
        output_path: Union[str, Path],
        overwrite: bool = True,
    ) -> Path:
        """
        Synthesize text and save audio output to a supported audio file (WAV).

        Args:
            text: Summary text to synthesize.
            output_path: Destination path for the audio file.
            overwrite: Whether to overwrite existing destination file (default: True).

        Returns:
            Resolved Path object pointing to the generated, non-empty audio file.

        Raises:
            ValueError / TypeError: On invalid text input or unsupported file extension.
            FileNotFoundError: If parent destination directory does not exist.
            FileExistsError: If destination file exists and overwrite=False.
            RuntimeError: If synthesis fails or generated file is missing/empty.
        """
        validated_text = self.validate_text(text)
        out_path = Path(output_path).resolve()

        # Validate file extension
        ext = out_path.suffix.lower()
        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported audio file format '{ext}'. "
                f"Supported formats for offline speech synthesis: {sorted(list(SUPPORTED_EXTENSIONS))}"
            )

        # Validate parent directory exists
        if not out_path.parent.exists():
            raise FileNotFoundError(
                f"Output directory does not exist: {out_path.parent}"
            )

        # Check overwrite policy
        if out_path.exists() and not overwrite:
            raise FileExistsError(
                f"Destination file already exists and overwrite is set to False: {out_path}"
            )

        # Use an injected engine (tests, pre-configured engines) as-is; otherwise
        # create a fresh engine instance for this synthesis, initializing COM on
        # the calling thread to avoid cross-thread blocking issues on Windows.
        try:
            if self._engine is not None:
                engine = self._engine
            else:
                if platform.system() == 'Windows':
                    try:
                        import pythoncom
                        pythoncom.CoInitialize()
                    except ImportError:
                        pass  # pythoncom may not be available in all environments
                engine = pyttsx3.init()
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize system speech engine: {exc}") from exc

        # Apply configuration
        try:
            engine.setProperty("rate", self._rate)
            engine.setProperty("volume", self._volume)
            if self._voice_id:
                engine.setProperty("voice", self._voice_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to configure speech engine properties: {exc}") from exc

        # Save to file and run synthesis
        try:
            engine.save_to_file(validated_text, str(out_path))
            engine.runAndWait()
        except Exception as exc:
            raise RuntimeError(f"Speech synthesis to file failed: {exc}") from exc
        finally:
            # Stop engines we created here; never stop an injected engine
            if engine is not self._engine:
                try:
                    engine.stop()
                except Exception:
                    pass

        # Verify that output file exists and is non-empty
        if not out_path.exists():
            raise RuntimeError(
                f"Speech synthesis reported completion, but destination file was not created: {out_path}"
            )

        file_size = out_path.stat().st_size
        if file_size == 0:
            raise RuntimeError(
                f"Speech synthesis failed: generated audio file is empty (0 bytes): {out_path}"
            )

        return out_path

    def stop(self) -> None:
        """Stop current speech synthesis or playback immediately."""
        if self._engine is not None:
            try:
                self._engine.stop()
            except Exception:
                pass


# =============================================================================
# CLI IMPLEMENTATION
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser for text-to-speech synthesis."""
    parser = argparse.ArgumentParser(
        description="Convert extractive news summaries into offline spoken audio using pyttsx3."
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="Summary text to speak or synthesize.",
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        default=None,
        help="Path to a text file containing the summary text.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination path for synthesized audio file (.wav).",
    )
    parser.add_argument(
        "--rate",
        type=int,
        default=DEFAULT_RATE,
        help=f"Speech rate in words per minute (range: {MIN_RATE}-{MAX_RATE}, default: {DEFAULT_RATE}).",
    )
    parser.add_argument(
        "--volume",
        type=float,
        default=DEFAULT_VOLUME,
        help=f"Speech volume (range: 0.0-1.0, default: {DEFAULT_VOLUME}).",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default=None,
        help="Voice ID to use for synthesis.",
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all available system voices and exit.",
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing destination audio files.",
    )
    return parser


def main() -> int:
    """CLI execution entrypoint."""
    parser = build_parser()
    args = parser.parse_args()

    try:
        engine = TextToSpeechEngine(
            rate=args.rate,
            volume=args.volume,
            voice_id=args.voice,
        )

        # Mode 1: List voices
        if args.list_voices:
            voices = engine.get_available_voices()
            print("\n" + "=" * 65)
            print("AVAILABLE SYSTEM VOICES")
            print("=" * 65)
            if not voices:
                print("No system voices detected.")
            for idx, v in enumerate(voices, 1):
                print(f"[{idx}] ID   : {v['id']}")
                print(f"    Name : {v['name']}")
                print(f"    Langs: {v['languages']}")
                print(f"    Gender: {v['gender']}")
                print("-" * 65)
            print()
            return 0

        # Retrieve text from argument or input file
        text_content: Optional[str] = None
        if args.text:
            text_content = args.text
        elif args.input_file:
            if not args.input_file.exists():
                print(f"[ERROR] Input file not found: {args.input_file}", file=sys.stderr)
                return 1
            text_content = args.input_file.read_text(encoding="utf-8")

        if not text_content:
            print(
                "[ERROR] Please provide text via --text '...' or --input-file <path>, "
                "or run with --list-voices to inspect available voices.",
                file=sys.stderr,
            )
            return 1

        # Mode 2: Save to audio file
        if args.output:
            overwrite = not args.no_overwrite
            out_file = engine.save_to_file(
                text=text_content,
                output_path=args.output,
                overwrite=overwrite,
            )
            print(f"[SUCCESS] Audio successfully synthesized and saved to: {out_file.resolve()}")
            print(f"[INFO] File size: {out_file.stat().st_size:,} bytes")
            return 0

        # Mode 3: Speak aloud directly
        print("[INFO] Speaking summary aloud via system audio output...")
        engine.speak(text_content)
        print("[SUCCESS] Speech playback completed.")
        return 0

    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
