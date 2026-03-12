"""
Tests for NLP Book Report assignment.

Tests saved artifacts in output/ folder.
"""
import os
import pytest


OUTPUT_DIR = "output"


def read_file(filename):
    """Read a file from the output directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    if not os.path.exists(path):
        pytest.fail(f"Missing required file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def test_characters_file_exists():
    """characters.txt was created."""
    assert os.path.exists(os.path.join(OUTPUT_DIR, "characters.txt")), \
        "Missing output/characters.txt"


def test_found_characters():
    """Found main characters (Holmes, Wilson, Watson)."""
    content = read_file("characters.txt").lower()
    found = any(name in content for name in ["holmes", "wilson", "watson"])
    assert found, "characters.txt should contain Holmes, Wilson, or Watson"


def test_locations_file_exists():
    """locations.txt was created."""
    assert os.path.exists(os.path.join(OUTPUT_DIR, "locations.txt")), \
        "Missing output/locations.txt"


def test_found_locations():
    """Found story locations (London, Fleet Street, Coburg Square)."""
    content = read_file("locations.txt").lower()
    found = any(place in content for place in ["london", "fleet street", "coburg"])
    assert found, "locations.txt should contain London, Fleet Street, or Coburg Square"


def test_business_file_exists():
    """business.txt was created."""
    assert os.path.exists(os.path.join(OUTPUT_DIR, "business.txt")), \
        "Missing output/business.txt"


def test_found_business():
    """Found Wilson's business (pawnbroker)."""
    content = read_file("business.txt").lower()
    assert "pawnbroker" in content, "business.txt should contain 'pawnbroker'"


def test_routine_file_exists():
    """routine.txt was created."""
    assert os.path.exists(os.path.join(OUTPUT_DIR, "routine.txt")), \
        "Missing output/routine.txt"


def test_found_work_routine():
    """Found Wilson's work routine (copying encyclopedia)."""
    content = read_file("routine.txt").lower()
    found = "copy" in content or "encyclop" in content
    assert found, "routine.txt should mention copying the encyclopedia"


def test_found_dissolution():
    """Found that the League was dissolved."""
    content = read_file("routine.txt").lower()
    assert "dissolve" in content, "routine.txt should mention the League was dissolved"
