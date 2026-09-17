"""
Spike/experiment (not part of the permanent agent/ codebase yet): verify
AssemblyAI's Dictation (live-upload) API actually transcribes our real
Hindi/Hinglish audio well before committing agent/stt.py to it.

Uses assemblyai==1.5.4's real DictationTranscriber API (introspected
directly rather than guessed from memory/docs -- see zdnd/decision.md):
transcribe_live() pushes a whole file over the same live-upload connection
used for streaming, returning a DictationResponse with .final_text.

Usage:
    python agent/stt_assemblyai_spike.py
"""
import os
import time

import assemblyai as aai
from dotenv import load_dotenv

load_dotenv()
aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]

SAMPLES = [
    ("english", "data/audio/public/en_00000.wav", "en"),
    ("hindi", "data/audio/public/hi_00000.wav", "hi"),
    ("hinglish (self-recorded, contains a name)", "data/audio/selfrecorded/selfrecorded_001.wav", "hi"),
    ("hinglish (self-recorded, no name)", "data/audio/selfrecorded/selfrecorded_066.wav", "hi"),
]


def main():
    transcriber = aai.DictationTranscriber()

    for label, path, lang_hint in SAMPLES:
        if not os.path.exists(path):
            print(f"{label}: skipping, {path} not found locally")
            continue
        config = aai.DictationConfig(language_codes=[lang_hint])
        start = time.perf_counter()
        result = transcriber.transcribe_live(path, config=config)
        elapsed = time.perf_counter() - start
        print(f"{label} ({path}, hint={lang_hint}):")
        print(f"  text:    {result.final_text}")
        print(f"  latency: {elapsed:.2f}s")
        print()

    # auto-detect, no hint, for comparison
    print("=== no language hint (auto-detect) ===")
    for label, path, _ in SAMPLES:
        if not os.path.exists(path):
            continue
        result = transcriber.transcribe_live(path)
        print(f"{label}: {result.final_text}")


if __name__ == "__main__":
    main()
