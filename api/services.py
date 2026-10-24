"""
Domain Services Layer
Phase 11: FastAPI Backend Integration

Connects and orchestrates existing NLP, vocabulary, evaluation, and speech
modules without modifying their underlying implementations.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import os
import subprocess
import sys
import uuid

from summarization.frequency import FrequencySummarizer
from summarization.tfidf import TFIDFSummarizer
from summarization.textrank import TextRankSummarizer
from vocabulary.extractor import VocabularyExtractor
from vocabulary.new_word_detector import NewWordDetector
from evaluation.rouge import evaluate_summary_pair
from speech.text_to_speech import TextToSpeechEngine


# Cache summarizer and extractor instances for performance across requests
_frequency_summarizer: Optional[FrequencySummarizer] = None
_tfidf_summarizer: Optional[TFIDFSummarizer] = None
_textrank_summarizer: Optional[TextRankSummarizer] = None
_vocabulary_extractor: Optional[VocabularyExtractor] = None


def get_frequency_summarizer() -> FrequencySummarizer:
    global _frequency_summarizer
    if _frequency_summarizer is None:
        _frequency_summarizer = FrequencySummarizer()
    return _frequency_summarizer


def get_tfidf_summarizer() -> TFIDFSummarizer:
    global _tfidf_summarizer
    if _tfidf_summarizer is None:
        _tfidf_summarizer = TFIDFSummarizer()
    return _tfidf_summarizer


def get_textrank_summarizer() -> TextRankSummarizer:
    global _textrank_summarizer
    if _textrank_summarizer is None:
        _textrank_summarizer = TextRankSummarizer()
    return _textrank_summarizer


def get_vocabulary_extractor() -> VocabularyExtractor:
    global _vocabulary_extractor
    if _vocabulary_extractor is None:
        _vocabulary_extractor = VocabularyExtractor()
    return _vocabulary_extractor


def run_summarization(
    text: str,
    method: str = "frequency",
    num_sentences: int = 3,
    article_id: str = "custom_article",
) -> Dict[str, Any]:
    """
    Run extractive summarization using the specified classical NLP algorithm.

    Args:
        text: Raw article text.
        method: One of 'frequency', 'tfidf', or 'textrank'.
        num_sentences: Number of summary sentences to extract.
        article_id: Identifier tag.

    Returns:
        Structured summarization dictionary.
    """
    clean_method = method.strip().lower()
    if clean_method == "frequency":
        summarizer = get_frequency_summarizer()
    elif clean_method == "tfidf":
        summarizer = get_tfidf_summarizer()
    elif clean_method == "textrank":
        summarizer = get_textrank_summarizer()
    else:
        raise ValueError(f"Unsupported summarization method: '{method}'. Must be 'frequency', 'tfidf', or 'textrank'.")

    return summarizer.summarize_text(
        raw_text=text,
        num_sentences=num_sentences,
        article_id=article_id,
    )


def run_vocabulary_extraction(
    text: str,
    article_id: str = "raw_article",
    max_items: int = 30,
    include_entities: bool = True,
) -> Dict[str, Any]:
    """
    Extract salient content words and named entities using Phase 7 VocabularyExtractor.
    """
    extractor = get_vocabulary_extractor()
    return extractor.extract_from_text(
        raw_text=text,
        article_id=article_id,
        max_items=max_items,
        include_entities=include_entities,
    )


def run_new_word_detection(
    extraction_record: Optional[Dict[str, Any]] = None,
    raw_text: Optional[str] = None,
    known_vocabulary: Optional[List[Union[str, Dict[str, Any]]]] = None,
    max_items: int = 30,
    include_entities: bool = True,
    article_id: str = "custom_article",
) -> Dict[str, Any]:
    """
    Detect new words against user-supplied known vocabulary using Phase 8 NewWordDetector.
    If raw_text is provided and extraction_record is None, extracts vocabulary first.
    """
    if extraction_record is None:
        if raw_text is None:
            raise ValueError("Either 'extraction_record' or 'text' must be provided.")
        extraction_record = run_vocabulary_extraction(
            text=raw_text,
            article_id=article_id,
            max_items=max_items,
            include_entities=include_entities,
        )

    detector = NewWordDetector(known_vocabulary=known_vocabulary or [])
    return detector.detect_new_words(extraction_record)


def run_rouge_evaluation(
    candidate_summary: str,
    reference_summary: str,
) -> Dict[str, Dict[str, float]]:
    """
    Compute ROUGE-1, ROUGE-2, and ROUGE-L scores using Phase 6 evaluation module.
    """
    return evaluate_summary_pair(candidate_summary, reference_summary)


# Hard ceiling for a single speech synthesis subprocess. SAPI/pyttsx3 can
# deadlock when engines are driven from multiple threads inside one process;
# running each synthesis in its own short-lived process makes that impossible
# and lets us enforce a timeout instead of hanging a request forever.
SYNTHESIS_TIMEOUT_SECONDS = 120


def run_speech_synthesis(
    text: str,
    audio_dir: Path,
    rate: int = 175,
    volume: float = 1.0,
    voice_id: Optional[str] = None,
    engine: Optional[TextToSpeechEngine] = None,
) -> Dict[str, Any]:
    """
    Synthesize text into speech and save audio to audio_dir.

    Executes `python -m speech.text_to_speech` in an isolated subprocess so a
    wedged COM/SAPI call can never block the API process; a failed or timed-out
    subprocess surfaces as RuntimeError (mapped to HTTP 500 by the router).
    """
    # Validate the requested voice cheaply in-process (no synthesis COM involved)
    # so unknown voices return HTTP 400 instead of a subprocess failure.
    if voice_id:
        validator = engine or TextToSpeechEngine(auto_init=False)
        available_ids = {v["id"] for v in validator.get_available_voices()}
        if available_ids and voice_id not in available_ids:
            raise ValueError(
                f"Requested voice ID '{voice_id}' is not available on this system."
            )

    audio_id = uuid.uuid4().hex
    file_name = f"{audio_id}.wav"
    output_path = audio_dir / file_name
    output_path.parent.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parent.parent
    cmd = [
        sys.executable,
        "-m",
        "speech.text_to_speech",
        "--text",
        text,
        "--output",
        str(output_path),
        "--rate",
        str(int(rate)),
        "--volume",
        str(float(volume)),
    ]
    if voice_id:
        cmd += ["--voice", voice_id]

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    try:
        completed = subprocess.run(
            cmd,
            cwd=str(project_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=SYNTHESIS_TIMEOUT_SECONDS,
            env=env,
        )
    except subprocess.TimeoutExpired as exc:
        output_path.unlink(missing_ok=True)
        raise RuntimeError(
            f"Speech synthesis timed out after {SYNTHESIS_TIMEOUT_SECONDS} seconds "
            "and was terminated."
        ) from exc

    failed = (
        completed.returncode != 0
        or not output_path.is_file()
        or output_path.stat().st_size == 0
    )
    if failed:
        stderr_lines = (completed.stderr or "").strip().splitlines()
        detail = stderr_lines[-1] if stderr_lines else f"exit code {completed.returncode}"
        output_path.unlink(missing_ok=True)
        raise RuntimeError(f"Speech synthesis failed: {detail}")

    file_size = output_path.stat().st_size

    return {
        "audio_id": audio_id,
        "audio_url": f"/api/v1/speech/audio/{audio_id}",
        "file_name": file_name,
        "file_size_bytes": file_size,
        "rate": rate,
        "volume": volume,
        "voice_id": voice_id,
    }
