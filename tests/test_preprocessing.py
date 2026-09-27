"""
Unit Tests for Phase 2: NLP Preprocessing Pipeline
Tests all 13 core requirements:
1. HTML cleaning
2. Unicode normalization
3. Whitespace normalization
4. Sentence segmentation
5. Tokenization
6. Lowercasing / normalization
7. Stop-word identification
8. Lemmatization
9. POS tagging
10. NER
11. Dual representation
12. JSONL output
13. Record-count validation and real CNN/DailyMail record integration
"""

import json
from pathlib import Path
import pytest
import spacy

from preprocessing.cleaner import clean_html
from preprocessing.normalizer import normalize_unicode, normalize_whitespace, normalize_token, normalize_text
from preprocessing.sentence_splitter import SentenceSplitter
from preprocessing.tokenizer import SpacyTokenizer
from preprocessing.stopwords import StopwordHandler
from preprocessing.lemmatizer import SpacyLemmatizer
from preprocessing.linguistic_features import LinguisticFeatureExtractor
from preprocessing.pipeline import PreprocessingPipeline


@pytest.fixture(scope="module")
def nlp_model():
    """Load spaCy model once for the test module."""
    return spacy.load("en_core_web_sm")


@pytest.fixture(scope="module")
def pipeline():
    """Initialize PreprocessingPipeline once."""
    return PreprocessingPipeline(model_name="en_core_web_sm")


# 1. HTML Cleaning Test
def test_clean_html():
    raw_html = "<p>President <b>announced</b> the decision&nbsp;today.</p><br>Next sentence here."
    cleaned = clean_html(raw_html)
    assert "<b>" not in cleaned
    assert "</b>" not in cleaned
    assert "<p>" not in cleaned
    assert "&nbsp;" not in cleaned
    assert "President announced the decision today." in cleaned
    assert "Next sentence here." in cleaned


# 2. Unicode Normalization Test
def test_unicode_normalization():
    # Decomposed e + acute accent (e\u0301) vs precomposed (é = \u00e9)
    decomposed = "re\u0301sume\u0301"
    normalized = normalize_unicode(decomposed, form="NFC")
    assert normalized == "résumé"
    assert len(normalized) == 6


# 3. Whitespace Normalization Test
def test_whitespace_normalization():
    raw_space = "This   is   a\n\n\nnews   article.\tWith   tabs."
    normalized = normalize_whitespace(raw_space)
    assert normalized == "This is a news article. With tabs."
    assert "  " not in normalized


# 4. Sentence Segmentation Test
def test_sentence_segmentation(nlp_model):
    splitter = SentenceSplitter(nlp=nlp_model)
    text = "The president announced the decision. The decision takes effect next month."
    sents = splitter.split_sentences(text)
    assert len(sents) == 2
    assert sents[0] == "The president announced the decision."
    assert sents[1] == "The decision takes effect next month."


# 5. Tokenization Test
def test_tokenization(nlp_model):
    tokenizer = SpacyTokenizer(nlp=nlp_model)
    sent = "The president announced the decision."
    tokens = tokenizer.tokenize(sent)
    assert tokens == ["The", "president", "announced", "the", "decision", "."]


# 6. Lowercasing / Normalization Test
def test_token_normalization():
    assert normalize_token("President") == "president"
    assert normalize_token("DECISION") == "decision"
    assert normalize_token("   Spaces   ") == "spaces"


# 7. Stop-word Identification Test
def test_stopword_identification(nlp_model):
    handler = StopwordHandler(nlp=nlp_model)
    assert handler.is_stopword("the") is True
    assert handler.is_stopword("and") is True
    assert handler.is_stopword("geopolitical") is False

    doc = nlp_model("The president announced the decision.")
    content_tokens = handler.filter_content_tokens(list(doc))
    # 'the' and '.' are removed; 'president', 'announced', 'decision' remain
    assert "the" not in content_tokens
    assert "." not in content_tokens
    assert "president" in content_tokens
    assert "announced" in content_tokens
    assert "decision" in content_tokens


# 8. Lemmatization Test
def test_lemmatization(nlp_model):
    lemmatizer = SpacyLemmatizer(nlp=nlp_model)
    doc = nlp_model("The cars are running and announced.")
    lemmas = [lemmatizer.lemmatize_token(tok) for tok in doc]
    assert "car" in lemmas  # cars -> car
    assert "run" in lemmas  # running -> run
    assert "announce" in lemmas  # announced -> announce


# 9. POS Tagging Test
def test_pos_tagging(nlp_model):
    extractor = LinguisticFeatureExtractor(nlp=nlp_model)
    doc = nlp_model("President speaks clearly.")
    pos_data = extractor.extract_pos_tags(list(doc))
    assert "pos" in pos_data
    assert "tags" in pos_data
    assert pos_data["pos"][0] in ["PROPN", "NOUN"]
    assert pos_data["pos"][1] in ["VERB"]
    assert pos_data["pos"][2] in ["ADV"]


# 10. NER Test
def test_ner_extraction(nlp_model):
    extractor = LinguisticFeatureExtractor(nlp=nlp_model)
    doc = nlp_model("Barack Obama visited Washington.")
    entities = extractor.extract_entities(doc)
    labels = {e["text"]: e["label"] for e in entities}
    assert "Barack Obama" in labels
    assert labels["Barack Obama"] == "PERSON"
    assert "Washington" in labels
    assert labels["Washington"] == "GPE"


# 11. Dual Representation Test
def test_dual_representation(pipeline):
    raw_article = "<p>The United Nations held talks in Geneva. Diplomats reached a new accord.</p>"
    result = pipeline.process_article(
        raw_article=raw_article,
        record_id="test_rec_01",
        highlights="Diplomats reach accord in Geneva.",
    )

    # A. Check original representation
    assert result["id"] == "test_rec_01"
    assert result["original_article"] == raw_article
    assert "<p>" not in result["cleaned_article"]
    assert len(result["original_sentences"]) == 2
    assert result["original_sentences"][0] == "The United Nations held talks in Geneva."

    # B. Check processed representation
    proc_sents = result["processed_sentences"]
    assert len(proc_sents) == 2
    sent0 = proc_sents[0]
    assert sent0["original_text"] == "The United Nations held talks in Geneva."
    assert "united" in sent0["normalized_tokens"]
    assert "geneva" in [l.lower() for l in sent0["lemmas"]]
    assert any(ent["text"] == "Geneva" for ent in sent0["entities"])
    assert any(ent["label"] == "GPE" for ent in sent0["entities"])


# 12. JSONL File Output Test
def test_jsonl_output(pipeline, tmp_path: Path):
    input_file = tmp_path / "sample_input.jsonl"
    output_file = tmp_path / "sample_output.jsonl"

    sample_records = [
        {"id": "rec_01", "article": "First article body text. Second sentence here.", "highlights": "Highlight 1"},
        {"id": "rec_02", "article": "Another article about technology. It works nicely.", "highlights": "Highlight 2"},
    ]
    with input_file.open("w", encoding="utf-8") as f:
        for r in sample_records:
            f.write(json.dumps(r) + "\n")

    count, stats = pipeline.process_file(input_file, output_file)
    assert count == 2
    assert output_file.exists()

    with output_file.open("r", encoding="utf-8") as f:
        lines = [json.loads(l) for l in f]
    assert len(lines) == 2
    assert lines[0]["id"] == "rec_01"
    assert len(lines[0]["original_sentences"]) == 2


# 13. Record-Count Validation and Real CNN/DailyMail Record Test
def test_real_cnn_dailymail_record(pipeline):
    raw_path = Path("dataset/raw/cnn_dailymail_test_100.jsonl")
    if not raw_path.exists():
        pytest.skip("dataset/raw/cnn_dailymail_test_100.jsonl not found")

    with raw_path.open("r", encoding="utf-8") as f:
        first_line = json.loads(f.readline())

    result = pipeline.process_article(
        raw_article=first_line["article"],
        record_id=first_line["id"],
        highlights=first_line["highlights"],
    )

    assert result["id"] == first_line["id"]
    assert len(result["original_sentences"]) > 0
    assert len(result["processed_sentences"]) == len(result["original_sentences"])
    # Check that sentences have token details and POS tags
    first_sent = result["processed_sentences"][0]
    assert len(first_sent["tokens"]) > 0
    assert len(first_sent["pos"]) == len(first_sent["tokens"])
    assert len(first_sent["lemmas"]) == len(first_sent["tokens"])


# 14. Tokenizer Parentheses & Dateline Regression Test
def test_tokenizer_parentheses_and_dateline(pipeline):
    """Verify (CNN)The is separated into distinct tokens without merging."""
    text = "(CNN)The Palestinian Authority"
    tokens = pipeline.tokenizer.tokenize(text)
    assert "(" in tokens
    assert "CNN" in tokens
    assert ")" in tokens
    assert "The" in tokens
    assert "CNN)The" not in tokens
    assert "(CNN)The" not in tokens


# 15. Tokenizer Punctuation Boundaries Regression Test
def test_tokenizer_punctuation_boundaries(pipeline):
    """Verify various punctuation boundary edge cases."""
    # Example 2: Adjacent quotes and commas
    t2 = pipeline.tokenizer.tokenize('word,"hello')
    assert "word" in t2
    assert "hello" in t2
    assert 'word,"hello' not in t2

    # Example 3: Standard quote and punctuation
    t3 = pipeline.tokenizer.tokenize('"Hello," said the reporter.')
    assert '"' in t3
    assert "Hello" in t3
    assert "," in t3
    assert "said" in t3

    # Example 4: Ordinary text
    t4 = pipeline.tokenizer.tokenize("The company announced the decision.")
    assert t4 == ["The", "company", "announced", "the", "decision", "."]

    # Example 5: Contractions
    t5 = pipeline.tokenizer.tokenize("don't")
    assert t5 == ["do", "n't"]

    # Example 6: Currency and numbers
    t6 = pipeline.tokenizer.tokenize("$100 million")
    assert t6 == ["$", "100", "million"]


# 16. NER Character Offsets Exact Alignment Test
def test_ner_character_offsets_exact_alignment(pipeline):
    """Verify that entity character offsets strictly match the sentence text substring."""
    article = "Barack Obama visited London on Wednesday, meeting with representatives of Microsoft."
    result = pipeline.process_article(article, record_id="ner_offset_test")
    assert len(result["processed_sentences"]) == 1
    sent = result["processed_sentences"][0]
    sent_text = sent["original_text"]
    entities = sent["entities"]
    assert len(entities) > 0

    for ent in entities:
        sliced_text = sent_text[ent["start_char"]:ent["end_char"]]
        assert sliced_text == ent["text"], (
            f"Offset mismatch: expected '{ent['text']}', got '{sliced_text}' "
            f"at offsets {ent['start_char']}..{ent['end_char']} in '{sent_text}'"
        )


# 17. Real Data Regression Test on CNN/DailyMail 1000 Sample
def test_real_dataset_1000_sample_hardening(pipeline):
    """Verify real CNN article has no malformed dateline tokens and has valid entity alignments."""
    sample_file = Path("dataset/raw/cnn_dailymail_test_1000.jsonl")
    if not sample_file.exists():
        pytest.skip("dataset/raw/cnn_dailymail_test_1000.jsonl not found")

    with sample_file.open("r", encoding="utf-8") as f:
        first_record = json.loads(f.readline())

    result = pipeline.process_article(
        raw_article=first_record["article"],
        record_id=first_record["id"],
        highlights=first_record["highlights"],
    )

    first_sent = result["processed_sentences"][0]
    tokens = first_sent["tokens"]

    # Assert CNN)The was not produced anywhere across the article
    all_tokens = [tok for s in result["processed_sentences"] for tok in s["tokens"]]
    assert "CNN)The" not in all_tokens
    assert "(CNN)The" not in all_tokens
    assert "CNN" in all_tokens
    assert ")" in all_tokens
    assert "The" in all_tokens

    # Assert all entity offsets in all sentences strictly align with sentence text
    for sent in result["processed_sentences"]:
        stext = sent["original_text"]
        for ent in sent["entities"]:
            assert stext[ent["start_char"]:ent["end_char"]] == ent["text"]

