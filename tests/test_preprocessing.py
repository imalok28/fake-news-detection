import os
import sys
import pytest

# Ensure project root is in sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocessing import clean_text

def test_clean_text_lowercasing():
    input_text = "The QUICK Brown Fox"
    expected = "quick brown fox"
    # Note: 'the' is a stopword and is removed
    assert clean_text(input_text, use_spacy=False) == expected

def test_clean_text_punctuation():
    input_text = "Hello, World!!! This is a test... with punctuation."
    # 'Hello', 'World', 'test', 'punctuation' remain (others like 'is', 'a', 'with' are stopwords)
    cleaned = clean_text(input_text, use_spacy=False)
    assert "," not in cleaned
    assert "!" not in cleaned
    assert "." not in cleaned
    assert "hello" in cleaned
    assert "world" in cleaned

def test_clean_text_stopwords():
    input_text = "this is an article containing various common english stopwords"
    cleaned = clean_text(input_text, use_spacy=False)
    # Stopwords like 'this', 'is', 'an', 'containing' (possibly), 'various', 'common', 'english' (possibly)
    # Let's verify 'this', 'is', 'an' are removed
    words = cleaned.split()
    assert "this" not in words
    assert "is" not in words
    assert "an" not in words

def test_clean_text_lemmatization():
    input_text = "running cars wolves studies"
    cleaned = clean_text(input_text, use_spacy=False)
    words = cleaned.split()
    # WordNetLemmatizer maps running -> running (for verb running, need tag, otherwise default is noun, so running stays running or run)
    # cars -> car
    # wolves -> wolf
    # studies -> study
    assert "car" in words
    assert "wolf" in words
    assert "study" in words

def test_clean_text_empty_and_none():
    assert clean_text("") == ""
    assert clean_text(None) == ""
    assert clean_text("   ") == ""
