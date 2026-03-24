"""
Unit tests for the rolling context window.
Pure logic, no external dependencies.
"""

import pytest

from pipeline.context_manager import ContextManager
from config import ContextConfig


def mgr(max_sentences=5) -> ContextManager:
    return ContextManager(ContextConfig(max_sentences=max_sentences))


# ── Tests ────────────────────────────────────────────────────────────────────

def test_empty_on_init():
    assert mgr().get_context() == ""
    assert len(mgr()) == 0


def test_single_sentence_returned():
    m = mgr()
    m.add("Hello world.")
    assert m.get_context() == "Hello world."


def test_multiple_sentences_space_joined():
    m = mgr()
    m.add("First.")
    m.add("Second.")
    assert m.get_context() == "First. Second."


def test_rolling_window_evicts_oldest():
    m = mgr(max_sentences=2)
    m.add("One.")
    m.add("Two.")
    m.add("Three.")          # evicts "One."
    ctx = m.get_context()
    assert "One." not in ctx
    assert "Two." in ctx
    assert "Three." in ctx


def test_len_tracks_sentence_count():
    m = mgr(max_sentences=3)
    assert len(m) == 0
    m.add("A.")
    assert len(m) == 1
    m.add("B.")
    m.add("C.")
    m.add("D.")              # evicts "A."
    assert len(m) == 3       # capped at max_sentences


def test_clear_empties_everything():
    m = mgr()
    m.add("Something.")
    m.clear()
    assert m.get_context() == ""
    assert len(m) == 0


def test_blank_and_whitespace_strings_ignored():
    m = mgr()
    m.add("")
    m.add("   ")
    m.add("\t\n")
    assert len(m) == 0
    assert m.get_context() == ""


def test_add_strips_whitespace():
    m = mgr()
    m.add("  Hello.  ")
    assert m.get_context() == "Hello."
