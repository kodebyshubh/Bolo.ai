"""
Fetch SPRINGLab/IndicTTS-English and SPRINGLab/IndicTTS-Hindi from Hugging Face,
filter each to a single consistent speaker, and export a ~1-2 hour subset as WAV
files (24kHz mono) into data/audio/public/, with matching text in
data/metadata_public.csv.

Uses streaming mode so we don't have to download the full multi-speaker corpus
just to pull out a couple hours of one speaker.

Usage:
    python data/fetch_public_corpus.py --hours 1.5
"""
import argparse
import csv
import io
import os
from collections import Counter
from itertools import islice

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate

SAMPLE_RATE = 24000
SPEAKER_SCAN_LIMIT = 5000  # rows scanned to find the most common speaker
SPEAKER_COL_CANDIDATES = ["speaker_id", "speaker", "spk_id", "speaker_name"]
TEXT_COL_CANDIDATES = ["text", "transcript", "sentence", "normalized_text"]

DATASETS = [
    ("SPRINGLab/IndicTTS-English", "en"),
    ("SPRINGLab/IndicTTS-Hindi", "hi"),
]


def find_column(columns, candidates, required=True):
    for c in candidates:
        if c in columns:
            return c
    if required:
        raise ValueError(f"none of {candidates} found in dataset columns: {sorted(columns)}")
    return None


def transliterate_hindi(text):
    """Devanagari -> ASCII Latin script (lowercased ITRANS).

    Verified locally against the actual orpheus-3b-0.1-ft tokenizer: raw
    Devanagari fragments into meaningless byte-level tokens on this
    Llama-3-based tokenizer (2.08 chars/token). Lowercased ITRANS tokenizes
    far more coherently (2.71 chars/token) than either raw Devanagari or
    mixed-case ITRANS (2.24) -- the mid-word capitals ITRANS uses for long
    vowels/retroflex consonants themselves confuse the tokenizer's subword
    boundaries. Still below English/Hinglish (3.1-3.9), but produces real
    subword tokens instead of byte garbage. See zdnd/decision.md.
    """
    itrans = transliterate(text, sanscript.DEVANAGARI, sanscript.ITRANS)
    itrans = itrans.replace("|", ".").replace("..", ".")
    return itrans.lower()


def pick_speaker(dataset_name, speaker_col):
    ds = load_dataset(dataset_name, split="train", streaming=True)
    counts = Counter(row[speaker_col] for row in islice(ds, SPEAKER_SCAN_LIMIT))
    speaker, n = counts.most_common(1)[0]
    print(f"{dataset_name}: sampled {sum(counts.values())} rows, speakers found: {dict(counts)}")
    print(f"{dataset_name}: picking most common speaker '{speaker}' ({n} rows in sample)")
    return speaker


def export_language(dataset_name, lang_tag, out_audio_dir, max_seconds):
    print(f"\n=== {dataset_name} ===")
    probe_ds = load_dataset(dataset_name, split="train", streaming=True)
    columns = list(probe_ds.features.keys())
    speaker_col = find_column(columns, SPEAKER_COL_CANDIDATES, required=False)
    text_col = find_column(columns, TEXT_COL_CANDIDATES)

    ds = load_dataset(dataset_name, split="train", streaming=True)
    # Decode audio ourselves via soundfile rather than the datasets library's
    # default decoder, which now requires the extra torchcodec dependency.
    ds = ds.cast_column("audio", Audio(decode=False))
    if speaker_col:
        chosen_speaker = pick_speaker(dataset_name, speaker_col)
        ds = ds.filter(lambda r: r[speaker_col] == chosen_speaker)
    else:
        print(f"{dataset_name}: no speaker column among {SPEAKER_COL_CANDIDATES}; assuming single-speaker dataset")

    os.makedirs(out_audio_dir, exist_ok=True)
    rows = []
    cumulative_seconds = 0.0
    for i, row in enumerate(ds):
        array, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        array = np.asarray(array, dtype=np.float32)
        duration = len(array) / sr

        if sr != SAMPLE_RATE:
            import librosa
            array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLE_RATE)
            sr = SAMPLE_RATE

        filename = f"{lang_tag}_{i:05d}.wav"
        sf.write(os.path.join(out_audio_dir, filename), array, sr, subtype="PCM_16")
        text = row[text_col]
        if lang_tag == "hi":
            text = transliterate_hindi(text)
        rows.append({"filename": f"public/{filename}", "text": text})

        cumulative_seconds += duration
        if cumulative_seconds >= max_seconds:
            break

    print(f"{dataset_name}: exported {len(rows)} clips, {cumulative_seconds / 3600:.2f} hours -> {out_audio_dir}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, default=1.5, help="target hours per language")
    parser.add_argument("--out-dir", default="data/audio/public", help="output audio directory")
    parser.add_argument("--metadata-out", default="data/metadata_public.csv", help="output metadata csv path")
    args = parser.parse_args()

    max_seconds = args.hours * 3600
    all_rows = []
    for dataset_name, lang_tag in DATASETS:
        all_rows += export_language(dataset_name, lang_tag, args.out_dir, max_seconds)

    os.makedirs(os.path.dirname(args.metadata_out) or ".", exist_ok=True)
    with open(args.metadata_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text"])
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nWrote {len(all_rows)} rows to {args.metadata_out}")


if __name__ == "__main__":
    main()
