"""
Interactive helper for the Phase 4 listening pass: plays a reference clip of
your own voice once (the actual fine-tuning target speaker), then for each
sentence plays the base and fine-tuned clips as blind "Clip A" / "Clip B"
(randomly ordered per row, per PRD section 8's blind-scoring method). Asks
for naturalness (1-5, standalone judgment), similarity (1-5, *to the
reference voice*, not to the other clip), and pronunciation_ok (0/1,
standalone) for each, then writes the answers back into the correct
base_/finetuned_ columns in results.csv.

Resumable: skips rows that already have scores filled in, and saves after
every row so an interruption doesn't lose progress.

Windows only (uses the stdlib winsound module for blocking WAV playback).

Usage:
    python evaluation/listen_and_score.py
"""
import csv
import random
import winsound

RESULTS_PATH = "evaluation/results.csv"
BASELINE_DIR = "evaluation/baseline_outputs"
FINETUNED_DIR = "evaluation/finetuned_outputs"
REFERENCE_VOICE_PATH = "data/audio/selfrecorded/selfrecorded_001.wav"

FIELDNAMES = [
    "id", "sentence", "language",
    "base_naturalness", "finetuned_naturalness",
    "base_similarity", "finetuned_similarity",
    "base_pronunciation_ok", "finetuned_pronunciation_ok",
]


def ask_int(prompt, valid):
    while True:
        raw = input(prompt).strip()
        if raw.isdigit() and int(raw) in valid:
            return int(raw)
        print(f"  enter one of {sorted(valid)}")


def play(path):
    winsound.PlaySound(path, winsound.SND_FILENAME)


def row_is_scored(row):
    return all(row[c].strip() != "" for c in FIELDNAMES[3:])


def main():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    remaining = [r for r in rows if not row_is_scored(r)]
    print(f"{len(rows)} total rows, {len(remaining)} left to score\n")

    if remaining:
        print("=== REFERENCE VOICE (target speaker -- your own recording) ===")
        print("Keep this voice in mind for every 'similarity' rating below:")
        print("similarity = how close each clip sounds to THIS voice, not to the other clip.")
        input("Press Enter to play the reference...")
        play(REFERENCE_VOICE_PATH)
        print()

    for row in remaining:
        sentence_id = row["id"]
        print(f"\n=== {sentence_id} ({row['language']}) ===")
        print(f"Text: {row['sentence']}")

        clips = ["base", "finetuned"]
        random.shuffle(clips)  # blind order -- you don't know which is which while scoring
        paths = {
            "base": f"{BASELINE_DIR}/{sentence_id}.wav",
            "finetuned": f"{FINETUNED_DIR}/{sentence_id}.wav",
        }

        answers = {}
        for slot, which in zip(["A", "B"], clips):
            input(f"Press Enter to play Clip {slot}...")
            play(paths[which])
            naturalness = ask_int(f"  Clip {slot} naturalness (1-5): ", set(range(1, 6)))
            similarity = ask_int(f"  Clip {slot} similarity to reference voice (1-5): ", set(range(1, 6)))
            pronunciation_ok = ask_int(f"  Clip {slot} pronunciation_ok (0 or 1): ", {0, 1})
            answers[which] = (naturalness, similarity, pronunciation_ok)

        row["base_naturalness"] = answers["base"][0]
        row["base_similarity"] = answers["base"][1]
        row["base_pronunciation_ok"] = answers["base"][2]
        row["finetuned_naturalness"] = answers["finetuned"][0]
        row["finetuned_similarity"] = answers["finetuned"][1]
        row["finetuned_pronunciation_ok"] = answers["finetuned"][2]

        with open(RESULTS_PATH, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved. ({rows.index(row) + 1}/{len(rows)} rows scored so far)")

    print("\nAll rows scored. Run: python evaluation/evaluate_tts.py --summarize-scores")


if __name__ == "__main__":
    main()
