"""
Sanity-check agent/tts.py: synthesize a brand-new sentence (not in the
Phase 2/4 test set) and confirm the model loads once and can be called
again without reloading. Needs a GPU -- run on Colab/Kaggle.

Usage:
    python agent/test_tts.py
"""
import os
import sys
import time

import soundfile as sf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.tts import SAMPLE_RATE, synthesize

NEW_SENTENCE = "Bolo ai apka naya voice assistant hai, kaise madad kar sakta hoon?"


def main():
    print("first call (includes model load):")
    start = time.perf_counter()
    audio = synthesize(NEW_SENTENCE)
    print(f"  {time.perf_counter() - start:.2f}s total")
    sf.write("agent_tts_test.wav", audio, SAMPLE_RATE)
    print("  wrote agent_tts_test.wav")

    print("second call (model already loaded, should be faster):")
    start = time.perf_counter()
    synthesize("Yeh dusra sentence hai.")
    print(f"  {time.perf_counter() - start:.2f}s total")


if __name__ == "__main__":
    main()
