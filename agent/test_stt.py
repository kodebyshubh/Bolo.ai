"""
Sanity-check agent/stt.py against a couple of real WAV files before wiring
it into the full pipeline: one English, one Hindi (skipped if not present
locally -- the public corpus is gitignored/reproduced via
data/fetch_public_corpus.py, not committed), and the self-recorded Hinglish
sample (always present, tracked in git).

Usage:
    python agent/test_stt.py
"""
import os
import sys

from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.stt import transcribe

load_dotenv()

SAMPLES = [
    ("english", "data/audio/public/en_00000.wav"),
    ("hindi", "data/audio/public/hi_00000.wav"),
    ("hinglish (self-recorded)", "data/audio/selfrecorded/selfrecorded_001.wav"),
]


def main():
    for label, path in SAMPLES:
        if not os.path.exists(path):
            print(f"{label}: skipping, {path} not found locally")
            continue
        result = transcribe(path)
        print(f"{label} ({path}):")
        print(f"  text:     {result['text']}")
        print(f"  language: {result['language']}")
        print()


if __name__ == "__main__":
    main()
