"""
Voice Activity Detection: identify speech vs silence in raw PCM audio, so
the agent doesn't wait through artificial dead air before/after the user
actually talks (the "artificial silence-padding delay" PRD v2 calls out),
ahead of eventually driving live mic capture in a later PRD v2 step.

Uses webrtcvad (installed here as webrtcvad-wheels, a prebuilt-binary
distribution of the same API -- plain webrtcvad needs a C compiler to
build from source, which this machine doesn't have).

Expects raw 16-bit mono PCM bytes (not a WAV file with its header still
attached) at one of webrtcvad's supported sample rates.
"""
import webrtcvad

FRAME_MS = 20  # webrtcvad only accepts 10, 20, or 30ms frames
SUPPORTED_SAMPLE_RATES = (8000, 16000, 32000, 48000)


def _frame_bytes(sample_rate, frame_ms=FRAME_MS):
    return int(sample_rate * frame_ms / 1000) * 2  # 16-bit mono = 2 bytes/sample


def frame_generator(pcm_bytes, sample_rate, frame_ms=FRAME_MS):
    """Split raw 16-bit mono PCM bytes into fixed-size frames webrtcvad can
    classify. Drops a trailing partial frame shorter than one full frame."""
    n = _frame_bytes(sample_rate, frame_ms)
    for offset in range(0, len(pcm_bytes) - n + 1, n):
        yield pcm_bytes[offset:offset + n]


def speech_flags(pcm_bytes, sample_rate, mode=2, frame_ms=FRAME_MS):
    """One bool per frame: True if that frame contains speech per
    webrtcvad. mode: 0 (least aggressive, most permissive about calling
    something speech) to 3 (most aggressive at rejecting non-speech)."""
    if sample_rate not in SUPPORTED_SAMPLE_RATES:
        raise ValueError(f"sample_rate must be one of {SUPPORTED_SAMPLE_RATES}, got {sample_rate}")
    vad = webrtcvad.Vad(mode)
    return [vad.is_speech(frame, sample_rate) for frame in frame_generator(pcm_bytes, sample_rate, frame_ms)]


def trim_silence(pcm_bytes, sample_rate, mode=2, frame_ms=FRAME_MS, padding_ms=100):
    """Trim leading/trailing silence from raw PCM audio, keeping a small
    padding around the detected speech so words aren't clipped. Returns the
    input unchanged if no speech is detected at all, rather than returning
    empty audio for a borderline-quiet clip."""
    n = _frame_bytes(sample_rate, frame_ms)
    flags = speech_flags(pcm_bytes, sample_rate, mode, frame_ms)
    if not any(flags):
        return pcm_bytes

    first_speech = flags.index(True)
    last_speech = len(flags) - 1 - flags[::-1].index(True)

    padding_frames = max(1, int(padding_ms / frame_ms))
    start_frame = max(0, first_speech - padding_frames)
    end_frame = min(len(flags) - 1, last_speech + padding_frames)

    return pcm_bytes[start_frame * n : (end_frame + 1) * n]


def find_end_of_speech(pcm_bytes, sample_rate, mode=2, frame_ms=FRAME_MS, silence_ms=500):
    """For a live/growing buffer: find the byte offset where sustained
    trailing silence begins, after at least one speech frame has occurred.
    Returns None if no such point exists yet in the given bytes (still
    speaking, or no speech detected at all) -- meant to be polled as more
    audio keeps arriving, to decide when the user has actually stopped
    talking rather than recording a fixed duration regardless."""
    n = _frame_bytes(sample_rate, frame_ms)
    flags = speech_flags(pcm_bytes, sample_rate, mode, frame_ms)
    silence_frames_needed = max(1, int(silence_ms / frame_ms))

    seen_speech = False
    consecutive_silence = 0
    for i, is_speech in enumerate(flags):
        if is_speech:
            seen_speech = True
            consecutive_silence = 0
        else:
            consecutive_silence += 1
            if seen_speech and consecutive_silence >= silence_frames_needed:
                end_frame = i - silence_frames_needed + 1
                return end_frame * n
    return None
