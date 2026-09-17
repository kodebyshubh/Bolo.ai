"""
Build data/metadata_selfrecorded.csv from a folder of self-recorded WAV files and
a transcript mapping file.

Transcript mapping file format: one entry per line, tab-separated:
    filename.wav<TAB>transcript text here <laugh>

Usage:
    python data/build_metadata.py --audio-dir data/audio/selfrecorded \
        --transcripts data/selfrecorded_transcripts.txt \
        --out data/metadata_selfrecorded.csv
"""
import argparse
import csv
import os
import re

EMOTION_TAG_RE = re.compile(r"<[a-zA-Z_]+>")
# after stripping emotion tags: letters (incl. Devanagari), digits, whitespace,
# and common sentence punctuation
ALLOWED_CHARS_RE = re.compile(r"^[a-zA-Z0-9ऀ-ॿ\s.,!?'\"-]*$")


def validate_transcript(text):
    stripped = EMOTION_TAG_RE.sub("", text)
    if ALLOWED_CHARS_RE.match(stripped):
        return True, []
    bad_chars = sorted({ch for ch in stripped if not ALLOWED_CHARS_RE.match(ch)})
    return False, bad_chars


def load_transcripts(path):
    mapping = {}
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(f"{path}:{lineno}: expected 'filename<TAB>text', got: {line!r}")
            filename, text = line.split("\t", 1)
            mapping[filename.strip()] = text.strip()
    return mapping


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio-dir", default="data/audio/selfrecorded")
    parser.add_argument("--transcripts", required=True, help="tab-separated filename->text mapping file")
    parser.add_argument("--out", default="data/metadata_selfrecorded.csv")
    args = parser.parse_args()

    transcripts = load_transcripts(args.transcripts)
    wav_files = {f for f in os.listdir(args.audio_dir) if f.lower().endswith(".wav")}

    missing_audio = sorted(set(transcripts) - wav_files)
    missing_transcript = sorted(wav_files - set(transcripts))
    if missing_audio:
        print(f"warning: {len(missing_audio)} transcripts have no matching WAV file: {missing_audio}")
    if missing_transcript:
        print(f"warning: {len(missing_transcript)} WAV files have no transcript: {missing_transcript}")

    rows = []
    for filename in sorted(set(transcripts) & wav_files):
        text = transcripts[filename]
        ok, bad_chars = validate_transcript(text)
        if not ok:
            print(f"warning: {filename}: transcript has unexpected characters {bad_chars}: {text!r}")
            continue
        rows.append({"filename": f"selfrecorded/{filename}", "text": text})

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
