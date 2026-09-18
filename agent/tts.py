"""
TTS component for the live agent. Loads the fine-tuned Orpheus model (base +
LoRA adapter) once, on first call, then reuses it for every synthesize()
call -- reloading per-call would add tens of seconds of model-load latency
to every single turn. Reuses inference/tts_server.py's model loading and
generation logic rather than duplicating it, so the agent and the offline
eval script can never drift apart on token scheme or decoding settings.

Needs a GPU -- same requirement as inference/tts_server.py.
"""
import os
import re
import sys

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.tts_server import SAMPLE_RATE
from inference.tts_server import load_models
from inference.tts_server import synthesize as _synthesize_with_models

DEFAULT_ADAPTER_PATH = "training/checkpoints/final_adapter"

# The fine-tuning data was romanized Hindi (ITRANS) and plain-ASCII
# punctuation only (see data/fetch_public_corpus.py, data/build_metadata.py's
# ALLOWED_CHARS_RE) -- native Devanagari script and "smart" typographic
# punctuation are both out-of-distribution for this checkpoint and were
# found, by ear, to produce unintelligible/gibberish audio. The LLM
# (agent/llm.py) is free to reply in either, so its output must be
# normalized to the training distribution before it reaches synthesize().
DEVANAGARI_RE = re.compile(r"[ऀ-ॿ]")
_PUNCT_NORMALIZE_MAP = str.maketrans({
    "‘": "'", "’": "'",  # ' '
    "“": '"', "”": '"',  # " "
    "–": "-", "—": "-", "‑": "-",  # – — U+2011 non-breaking hyphen
    "(": " ", ")": " ",
})
# Anything outside this set was never in the training data (see
# data/build_metadata.py's own ALLOWED_CHARS_RE for the self-recorded side)
# -- strip it rather than let the model try to pronounce an unseen
# character. Applied after transliteration, so no Devanagari should remain
# by this point; kept ASCII-only intentionally. Replaced with a space, not
# deleted outright -- deleting can fuse two real words into one nonsense
# word the model never saw (e.g. an unhandled Unicode hyphen variant once
# turned "ice-cube" into "icecube"), which is worse than leaving a gap.
_DISALLOWED_CHARS_RE = re.compile(r"[^a-zA-Z0-9\s.,!?'\"-]")
_MULTISPACE_RE = re.compile(r"\s+")

# The self-recorded training script (data/hinglish_recording_script.txt)
# always speaks numbers digit-by-digit ("order number one two three four
# five", "pincode four one one zero one four") -- raw numeral characters
# like "12345" or a bare list marker "2." were never seen as digits during
# fine-tuning and were found to produce gibberish. Convert digit-by-digit,
# matching the established training convention, rather than to
# magnitude words ("twelve thousand...") which the data never modeled either.
_DIGIT_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")
_DIGITS_RE = re.compile(r"\d+")


def _digits_to_words(match):
    return " ".join(_DIGIT_WORDS[int(d)] for d in match.group(0))


def _sanitize_for_tts(text):
    if DEVANAGARI_RE.search(text):
        # the transliteration library renders the Devanagari danda (।) as a
        # literal '|' pipe character, which is itself out-of-distribution --
        # same fix data/fetch_public_corpus.py's transliterate_hindi() applies.
        text = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
        text = text.replace("|", ".").replace("..", ".")
        text = text.lower()
    text = text.translate(_PUNCT_NORMALIZE_MAP)
    text = _DIGITS_RE.sub(_digits_to_words, text)
    text = _DISALLOWED_CHARS_RE.sub(" ", text)
    return _MULTISPACE_RE.sub(" ", text).strip()

# Emotion tags the fine-tuning data actually used (see
# data/hinglish_recording_script.txt section E) -- an unrecognized tag
# wasn't seen during training, so its effect on generation is unverified.
KNOWN_EMOTION_TAGS = {"laugh", "sigh", "angry", "excited"}
EMOTION_TAG_RE = re.compile(r"<([a-zA-Z_]+)>")

_model = None
_tokenizer = None
_snac_model = None


def _check_emotion_tags(text):
    for tag in EMOTION_TAG_RE.findall(text):
        if tag.lower() not in KNOWN_EMOTION_TAGS:
            print(f"agent.tts: warning: <{tag}> was not in the fine-tuning data's emotion tags {sorted(KNOWN_EMOTION_TAGS)} -- effect on output is unverified")


def _ensure_loaded(adapter_path):
    global _model, _tokenizer, _snac_model
    if _model is not None:
        return
    path = adapter_path if adapter_path and os.path.exists(adapter_path) else None
    if adapter_path and path is None:
        print(f"agent.tts: warning: adapter path {adapter_path!r} not found, falling back to base model")
    print(f"agent.tts: loading model (adapter={path or 'none -- base model'})...")
    _model, _tokenizer, _snac_model = load_models(adapter_path=path)
    print("agent.tts: model loaded.")


def synthesize(text: str, voice=None, adapter_path=DEFAULT_ADAPTER_PATH):
    """Synthesize `text` to a numpy float32 waveform at SAMPLE_RATE. Loads
    the model on the first call only; subsequent calls reuse it.

    Emotion tags like <laugh>/<sigh> are passed straight through to the
    model as-is (Orpheus was trained to interpret them inline) -- this just
    warns if a tag outside the fine-tuning data's known set is used, since
    its effect on generation wasn't verified during training."""
    text = _sanitize_for_tts(text)
    print(f"agent.tts: sanitized text sent to model: {text!r}")
    _check_emotion_tags(text)
    _ensure_loaded(adapter_path)
    return _synthesize_with_models(_model, _tokenizer, _snac_model, text, voice=voice)
