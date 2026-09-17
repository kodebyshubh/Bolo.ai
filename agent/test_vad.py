"""
Tests agent/vad.py against synthetic silence/tone PCM audio (no real
recording or GPU needed -- pure signal generation and VAD classification).

Usage:
    python agent/test_vad.py
"""
import math
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.vad import find_end_of_speech, speech_flags, trim_silence

SAMPLE_RATE = 16000


def silence_ms(ms):
    n_samples = int(SAMPLE_RATE * ms / 1000)
    return b"\x00\x00" * n_samples


def tone_ms(ms, freq=200):
    n_samples = int(SAMPLE_RATE * ms / 1000)
    samples = [int(10000 * math.sin(2 * math.pi * freq * i / SAMPLE_RATE)) for i in range(n_samples)]
    return b"".join(struct.pack("<h", s) for s in samples)


def test_speech_flags_distinguishes_silence_and_tone():
    audio = silence_ms(300) + tone_ms(300) + silence_ms(300)
    flags = speech_flags(audio, SAMPLE_RATE)
    n_frames = len(flags)
    # roughly first third silent, middle third speech, last third silent
    third = n_frames // 3
    print(f"test_speech_flags: {n_frames} frames, flags={flags}")
    assert not any(flags[:third - 1]), "expected leading frames to be non-speech"
    assert any(flags[third:2 * third]), "expected middle frames to contain speech"
    print("  PASS\n")


def test_trim_silence_removes_leading_and_trailing_silence():
    audio = silence_ms(500) + tone_ms(500) + silence_ms(500)
    trimmed = trim_silence(audio, SAMPLE_RATE, padding_ms=0)
    print(f"test_trim_silence: original {len(audio)} bytes -> trimmed {len(trimmed)} bytes")
    assert len(trimmed) < len(audio), "expected trimming to shrink the audio"
    assert len(trimmed) < len(audio) * 0.7, "expected most of the 1000ms of silence to be removed"
    print("  PASS\n")


def test_trim_silence_returns_unchanged_when_no_speech():
    audio = silence_ms(500)
    trimmed = trim_silence(audio, SAMPLE_RATE)
    print(f"test_trim_silence_no_speech: {len(audio)} -> {len(trimmed)} bytes")
    assert trimmed == audio, "expected all-silence input to be returned unchanged"
    print("  PASS\n")


def test_find_end_of_speech_detects_trailing_pause():
    # speak for 500ms, then go silent for 800ms (well over the 500ms
    # threshold) -- find_end_of_speech should report a point shortly after
    # the tone ends, not None and not somewhere inside the tone itself
    audio = tone_ms(500) + silence_ms(800)
    end_offset = find_end_of_speech(audio, SAMPLE_RATE, silence_ms=500)
    print(f"test_find_end_of_speech: audio len={len(audio)} bytes, end_offset={end_offset}")
    assert end_offset is not None, "expected a detected end-of-speech point"
    tone_byte_len = len(tone_ms(500))
    assert tone_byte_len * 0.8 <= end_offset <= tone_byte_len * 1.3, (
        f"expected end_offset near the end of the 500ms tone ({tone_byte_len} bytes), got {end_offset}"
    )
    print("  PASS\n")


def test_find_end_of_speech_none_while_still_speaking():
    # a short pause that doesn't reach the silence threshold shouldn't
    # trigger end-of-speech
    audio = tone_ms(300) + silence_ms(100) + tone_ms(300)
    end_offset = find_end_of_speech(audio, SAMPLE_RATE, silence_ms=500)
    print(f"test_find_end_of_speech_none: end_offset={end_offset}")
    assert end_offset is None, "expected no end-of-speech point for a brief mid-sentence pause"
    print("  PASS\n")


def main():
    test_speech_flags_distinguishes_silence_and_tone()
    test_trim_silence_removes_leading_and_trailing_silence()
    test_trim_silence_returns_unchanged_when_no_speech()
    test_find_end_of_speech_detects_trailing_pause()
    test_find_end_of_speech_none_while_still_speaking()
    print("all tests passed")


if __name__ == "__main__":
    main()
