"""
Speech-to-text via AssemblyAI's Dictation API (see zdnd/decision.md for why
Groq's Whisper endpoint was replaced -- it's confirmed batch-only, no
streaming support at all, whereas AssemblyAI's Dictation API uploads audio
progressively as it's produced).

Never uses unrestricted auto-detect: it was found to occasionally
misdetect short Hinglish utterances as an unrelated language entirely (one
test sample came back as Mandarin Chinese). But a single *forced* language
hint is also wrong for this agent, which is meant to serve English, Hindi,
and Hinglish speakers -- forcing "hi" degrades genuine English audio into
phonetic Devanagari nonsense. The default here restricts the language set
to exactly the two this agent supports (language_codes=["en", "hi"])
rather than forcing one or leaving it fully open: verified against real
audio to correctly transcribe English, Hindi, and (this is the interesting
part) Hinglish in its natural mixed script -- e.g. "Hi, मैं ... रहा हूँ"
keeping the English greeting in Latin script instead of forcing the whole
utterance into one alphabet.

Requires ASSEMBLYAI_API_KEY in the environment (see .env.example).
"""
import os

import assemblyai as aai

DEFAULT_LANGUAGE_CODES = ["en", "hi"]

_configured = False


def _ensure_configured():
    global _configured
    if not _configured:
        aai.settings.api_key = os.environ["ASSEMBLYAI_API_KEY"]
        _configured = True


def transcribe(audio, language=None):
    """audio: a file path (str/Path), or raw audio bytes.
    language: an explicit ISO-639-1 code ("en" or "hi") to force a single
    language; omit to use the restricted-set default (see module
    docstring) that lets AssemblyAI pick between English and Hindi per
    utterance, including within a single code-switched utterance.

    Returns {"text": str, "language": str} -- language is the single code
    when one was forced, or "en/hi" when the restricted-set default was
    used (a single Dictation call doesn't report back which of the allowed
    languages it actually matched)."""
    _ensure_configured()
    language_codes = [language] if language else DEFAULT_LANGUAGE_CODES

    transcriber = aai.DictationTranscriber()
    config = aai.DictationConfig(language_codes=language_codes)
    source = audio if isinstance(audio, (bytes, bytearray)) else str(audio)
    result = transcriber.transcribe_live(source, config=config)

    return {"text": result.final_text.strip(), "language": language or "/".join(language_codes)}
