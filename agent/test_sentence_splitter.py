"""
Tests agent/sentence_splitter.py against real LLM outputs already observed
in this project (Phase 6/8 testing) plus synthetic edge cases (decimals,
ellipsis), and confirms the result is identical regardless of how the input
is chunked -- since a real streamed API response can fragment text at
arbitrary boundaries, not just on word breaks.

Usage:
    python agent/test_sentence_splitter.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.stdout.reconfigure(encoding="utf-8")
from agent.sentence_splitter import split_sentences

CASES = [
    (
        "Hinglish, real output from Phase 6/8 testing",
        "Aapka order abhi processing mein hai. Expected delivery 25 September 2026 ko hai. 😊",
        [
            # trailing lone emoji is correctly dropped -- nothing to
            # pronounce, so it's not expected in the output.
            "Aapka order abhi processing mein hai.",
            "Expected delivery 25 September 2026 ko hai.",
        ],
    ),
    (
        "pure Hindi with danda, real output from Phase 6 testing",
        "ऐसा लगता है कि ऑर्डर 99999 हमारे सिस्टम में नहीं मिला। कृपया ऑर्डर नंबर दोबारा जाँचें या हमें सही विवरण दें, ताकि हम मदद कर सकें।",
        [
            "ऐसा लगता है कि ऑर्डर 99999 हमारे सिस्टम में नहीं मिला।",
            "कृपया ऑर्डर नंबर दोबारा जाँचें या हमें सही विवरण दें, ताकि हम मदद कर सकें।",
        ],
    ),
    (
        "decimal number should not split mid-number",
        "Your total is 12.50 dollars. Thanks!",
        ["Your total is 12.50 dollars.", "Thanks!"],
    ),
    (
        "ellipsis: split at the first period, extra dots dropped (no word content to preserve for TTS)",
        "Wait... let me check that.",
        ["Wait.", "let me check that."],
    ),
]


def run_case(label, text, expected):
    # test with three different chunkings: whole string, word-by-word,
    # character-by-character -- result must be identical regardless
    chunkings = {
        "whole": [text],
        "word-by-word": text.split(" "),
        "char-by-char": list(text),
    }
    results = {}
    for chunk_label, chunks in chunkings.items():
        if chunk_label == "word-by-word":
            chunks = [c + " " for c in chunks[:-1]] + [chunks[-1]]
        results[chunk_label] = list(split_sentences(chunks))

    print(f"=== {label} ===")
    for chunk_label, result in results.items():
        print(f"  [{chunk_label}] {result}")

    all_same = len(set(tuple(r) for r in results.values())) == 1
    print(f"  consistent across chunkings: {all_same}")
    if expected is not None:
        matches = results["whole"] == expected
        print(f"  matches expected: {matches}")
        if not matches:
            print(f"  EXPECTED: {expected}")
    print()


def main():
    for label, text, expected in CASES:
        run_case(label, text, expected)


if __name__ == "__main__":
    main()
