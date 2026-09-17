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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from inference.tts_server import SAMPLE_RATE
from inference.tts_server import load_models
from inference.tts_server import synthesize as _synthesize_with_models

DEFAULT_ADAPTER_PATH = "training/checkpoints/final_adapter"

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
    _check_emotion_tags(text)
    _ensure_loaded(adapter_path)
    return _synthesize_with_models(_model, _tokenizer, _snac_model, text, voice=voice)
