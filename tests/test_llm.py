"""
Integration tests for the LLM rewriter.

Requires: Ollama running with dolphin3 (or another model) available.
Tests that need Ollama are auto-skipped if it's unreachable.
"""

import pytest

from llm.rewriter import LLMRewriter
from config import LLMConfig


@pytest.fixture(scope="module")
def rewriter():
    r = LLMRewriter(LLMConfig(model="dolphin3"))
    r.load()
    return r


def skip_if_offline(rewriter):
    if not rewriter.config.enabled:
        pytest.skip("Ollama not reachable — skipping LLM test")


# ── Tests ────────────────────────────────────────────────────────────────────

def test_empty_input_returns_empty(rewriter):
    """Empty string must pass through without calling Ollama."""
    result = rewriter.rewrite("")
    assert result == ""


def test_whitespace_input_returns_empty(rewriter):
    result = rewriter.rewrite("   ")
    assert result == ""


def test_basic_cleanup(rewriter):
    """Filler words and bad punctuation should be cleaned up."""
    skip_if_offline(rewriter)
    result = rewriter.rewrite("uh so like i wanted to say hello um to everyone")
    assert isinstance(result, str)
    assert len(result) > 5
    # The LLM should not echo the system prompt back
    assert "dictation assistant" not in result.lower()
    assert "rules:" not in result.lower()


def test_returns_string_not_prompt(rewriter):
    """LLM must not return the prompt template or system message."""
    skip_if_offline(rewriter)
    result = rewriter.rewrite("this is a test")
    assert "Raw speech input" not in result
    assert "Context" not in result
    assert "Cleaned text" not in result


def test_context_passed_without_crash(rewriter):
    """Passing context must not crash and must still return a string."""
    skip_if_offline(rewriter)
    result = rewriter.rewrite(
        transcript="we should ship it next week",
        context="We discussed the product release schedule yesterday.",
    )
    assert isinstance(result, str)
    assert len(result) > 5


def test_no_context_works(rewriter):
    """No context (empty string) must work as a first utterance."""
    skip_if_offline(rewriter)
    result = rewriter.rewrite("hello this is the first thing i said", context="")
    assert isinstance(result, str)
    assert len(result) > 0


def test_already_clean_input_returned(rewriter):
    """Clean, well-punctuated input should come back roughly unchanged."""
    skip_if_offline(rewriter)
    clean = "The meeting is scheduled for Thursday at 3 PM."
    result = rewriter.rewrite(clean)
    # Should not be radically different from input
    # Check key words are preserved
    assert "Thursday" in result or "thursday" in result.lower()
    assert "3" in result or "three" in result.lower()


def test_fallback_on_disabled(rewriter):
    """When LLM is disabled, rewrite must return the raw transcript."""
    r = LLMRewriter(LLMConfig(enabled=False))
    raw = "this is the raw transcript"
    assert r.rewrite(raw) == raw
