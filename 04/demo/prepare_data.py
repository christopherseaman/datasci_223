"""
One-off script to fetch Project Gutenberg texts and write curated excerpts
to data/ for Lecture 04 NLP demos. Output paths and filenames come from
config.yaml. Run from lectures/04/demo/.
"""

import re
import urllib.request
from pathlib import Path

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_PATH = SCRIPT_DIR / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DATA_DIR = SCRIPT_DIR / CONFIG["data"]["dir"]
GUTENBERG_BASE = "https://www.gutenberg.org/files"

# Optional: mirror that allows server identification
REQUEST_HEADERS = {"User-Agent": "Python-urllib (UCSF DataSci 223 demo)"}


def fetch_text(book_id: int) -> str:
    url = f"{GUTENBERG_BASE}/{book_id}/{book_id}-0.txt"
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def strip_gutenberg_boilerplate(text: str) -> str:
    start_marker = "*** START OF"
    end_marker = "*** END OF"
    if start_marker in text:
        text = text.split(start_marker, 1)[1]
        if "***" in text:
            text = text.split("***", 1)[1]
    if end_marker in text:
        text = text.split(end_marker)[0]
    return text.strip()


def normalize_whitespace(text: str) -> str:
    """Single newlines between paragraphs; strip trailing/leading per line."""
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def word_count(text: str) -> int:
    return len(text.split())


def extract_alice_ch1(full: str) -> str:
    full = strip_gutenberg_boilerplate(full)
    # Chapter 1: from "CHAPTER I." title through end of chapter (before CHAPTER II.)
    start = re.search(
        r"CHAPTER I\.\s*\nDown the Rabbit-Hole\s*\n", full, re.IGNORECASE
    )
    if not start:
        start = re.search(r"CHAPTER I\.\s*\n", full)
    if not start:
        raise ValueError("Could not find start of Chapter I")
    chunk = full[start.end() :]
    end = re.search(r"\n\s*CHAPTER II\.", chunk)
    if end:
        chunk = chunk[: end.start()]
    return normalize_whitespace(chunk)


def extract_holmes_scandal(full: str, max_words: int = 4200) -> str:
    full = strip_gutenberg_boilerplate(full)
    # Adventures of Sherlock Holmes: first story is "A Scandal in Bohemia"
    start = re.search(r"A SCANDAL IN BOHEMIA\s*\n", full)
    if not start:
        start = re.search(r"I\.\s*\n\s*A Scandal in Bohemia\s*\n", full)
    if not start:
        raise ValueError("Could not find start of A Scandal in Bohemia")
    chunk = full[start.end() :]
    # Stop at next story (II. or next title) or at max_words
    end = re.search(r"\n\s*II\.\s*\n|\n\s*THE RED-HEADED LEAGUE\s*\n", chunk)
    if end:
        chunk = chunk[: end.start()]
    words = chunk.split()
    if len(words) > max_words:
        chunk = " ".join(words[:max_words])
    return normalize_whitespace(chunk)


def extract_pride_ch1_ch2(full: str) -> str:
    full = strip_gutenberg_boilerplate(full)
    # Pride and Prejudice (Gutenberg 1342): uses Roman numerals; Ch I-II before Ch III
    # Narrative starts after "[Illustration: ... Chapter I.]" with "It is a truth..."
    start_m = re.search(
        r"(?:Chapter I\.|CHAPTER I\.)[^\n]*\]\s*\n\s*(It is a truth universally acknowledged)",
        full,
        re.IGNORECASE | re.DOTALL,
    )
    if not start_m:
        start_m = re.search(r"CHAPTER I\.\s*\n", full)
        if start_m:
            start_pos = start_m.end()
        else:
            raise ValueError("Could not find start of Chapter I")
    else:
        start_pos = start_m.start(1)
    chunk = full[start_pos:]
    end = re.search(
        r"\n\s*CHAPTER III\.|\n\s*Chapter III\.", chunk, re.IGNORECASE
    )
    if end:
        chunk = chunk[: end.start()]
    return normalize_whitespace(chunk)


def extract_frankenstein_ch5(full: str) -> str:
    full = strip_gutenberg_boilerplate(full)
    # Gutenberg 84 has a TOC at the start ("Chapter 5\n Chapter 6\n..."); body has "Chapter 5\n\n\nIt was on a dreary..."
    start_m = re.search(
        r"Chapter 5\s*\n\s*\n\s*(It was on a dreary night of November)",
        full,
        re.IGNORECASE,
    )
    if not start_m:
        start = re.search(r"Chapter 5\s*\n|CHAPTER 5\s*\n", full, re.IGNORECASE)
        if not start:
            raise ValueError("Could not find start of Chapter 5")
        chunk = full[start.end() :]
    else:
        chunk = full[start_m.start(1) :]
    # End at body Chapter 6 (unique follow-up "Clerval then")
    end = re.search(r"\n\s*Chapter 6\s*\n\s*\n\s*Clerval", chunk, re.IGNORECASE)
    if not end:
        end = re.search(
            r"\n\s*Chapter 6\s*\n|\n\s*CHAPTER 6\s*\n", chunk, re.IGNORECASE
        )
    if end:
        chunk = chunk[: end.start()]
    return normalize_whitespace(chunk)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    files = CONFIG["data"]["files"]

    # Alice in Wonderland (Ch 1) ~2,000 words
    text = fetch_text(11)
    excerpt = extract_alice_ch1(text)
    out_path = DATA_DIR / files["alice"]
    out_path.write_text(excerpt, encoding="utf-8")
    print(f"{out_path.name}: {word_count(excerpt)} words")

    # Sherlock Holmes, A Scandal in Bohemia ~4,000 words
    text = fetch_text(1661)
    excerpt = extract_holmes_scandal(text)
    out_path = DATA_DIR / files["holmes"]
    out_path.write_text(excerpt, encoding="utf-8")
    print(f"{out_path.name}: {word_count(excerpt)} words")

    # Pride and Prejudice (Ch 1-2) ~3,000 words
    text = fetch_text(1342)
    excerpt = extract_pride_ch1_ch2(text)
    out_path = DATA_DIR / files["pride"]
    out_path.write_text(excerpt, encoding="utf-8")
    print(f"{out_path.name}: {word_count(excerpt)} words")

    # Frankenstein (Ch 5) ~2,500 words
    text = fetch_text(84)
    excerpt = extract_frankenstein_ch5(text)
    out_path = DATA_DIR / files["frankenstein"]
    out_path.write_text(excerpt, encoding="utf-8")
    print(f"{out_path.name}: {word_count(excerpt)} words")


if __name__ == "__main__":
    main()
