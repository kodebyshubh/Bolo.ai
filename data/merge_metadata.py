"""
Merge data/metadata_public.csv and data/metadata_selfrecorded.csv into a single
data/metadata.csv, verifying every referenced audio file actually resolves under
data/audio/.

Usage:
    python data/merge_metadata.py
"""
import argparse
import csv
import os


def read_csv(path):
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", default="data/metadata_public.csv")
    parser.add_argument("--selfrecorded", default="data/metadata_selfrecorded.csv")
    parser.add_argument("--audio-root", default="data/audio")
    parser.add_argument("--out", default="data/metadata.csv")
    args = parser.parse_args()

    rows = []
    for path in (args.public, args.selfrecorded):
        if os.path.exists(path):
            rows += read_csv(path)
        else:
            print(f"note: {path} not found, skipping")

    missing = [r["filename"] for r in rows if not os.path.exists(os.path.join(args.audio_root, r["filename"]))]
    if missing:
        print(f"warning: {len(missing)} rows reference audio files that don't resolve under {args.audio_root}:")
        for f in missing[:20]:
            print(f"  {f}")
        if len(missing) > 20:
            print(f"  ... and {len(missing) - 20} more")

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "text"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.out} ({len(rows) - len(missing)} resolve, {len(missing)} missing)")


if __name__ == "__main__":
    main()
