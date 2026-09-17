"""
Compare base vs fine-tuned Orpheus output for Phase 4.

Two modes:
  --summarize-latency  reads baseline_latency.csv + finetuned_latency.csv,
                        prints average/median generation time per model.
  --build-scoring       builds evaluation/results.csv, a scoring template
                        (sentence/language pre-filled, score columns blank)
                        for you to fill in by listening to both sets side by
                        side (per PRD section 8's blind-scoring method).
  --summarize-scores    once results.csv is filled in, averages each score
                        column and prints a results table.

Usage:
    python evaluation/evaluate_tts.py --summarize-latency
    python evaluation/evaluate_tts.py --build-scoring
    python evaluation/evaluate_tts.py --summarize-scores
"""
import argparse
import csv
import statistics


def summarize_latency(baseline_path, finetuned_path):
    for label, path in [("baseline", baseline_path), ("finetuned", finetuned_path)]:
        with open(path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        times = [float(r["generation_seconds"]) for r in rows]
        if not times:
            print(f"{label}: no rows in {path}")
            continue
        print(f"{label} ({len(times)} sentences):")
        print(f"  mean:   {statistics.mean(times):.2f}s")
        print(f"  median: {statistics.median(times):.2f}s")
        print(f"  min:    {min(times):.2f}s")
        print(f"  max:    {max(times):.2f}s")
        print()
    print("--summarize-latency complete.")


def build_scoring_template(sentences_path, out_path):
    with open(sentences_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    fieldnames = [
        "id", "sentence", "language",
        "base_naturalness", "finetuned_naturalness",
        "base_similarity", "finetuned_similarity",
        "base_pronunciation_ok", "finetuned_pronunciation_ok",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "id": row["id"],
                "sentence": row["sentence"],
                "language": row["language"],
                "base_naturalness": "",
                "finetuned_naturalness": "",
                "base_similarity": "",
                "finetuned_similarity": "",
                "base_pronunciation_ok": "",
                "finetuned_pronunciation_ok": "",
            })
    print(f"wrote {len(rows)}-row scoring template to {out_path}")
    print("naturalness/similarity: rate 1-5 by ear (blind if possible -- don't look at which column is which while listening)")
    print("pronunciation_ok: 1 (correct) or 0 (mispronounced/wrong), especially for names/Hindi words/code-switched sentences")
    print("--build-scoring complete.")


def summarize_scores(results_path):
    with open(results_path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    score_cols = [
        "base_naturalness", "finetuned_naturalness",
        "base_similarity", "finetuned_similarity",
        "base_pronunciation_ok", "finetuned_pronunciation_ok",
    ]
    print(f"{len(rows)} rows in {results_path}\n")
    for col in score_cols:
        values = [float(r[col]) for r in rows if r[col].strip() != ""]
        if not values:
            print(f"{col}: no scores filled in yet")
            continue
        print(f"{col}: mean {statistics.mean(values):.2f} ({len(values)}/{len(rows)} scored)")
    print("--summarize-scores complete.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-latency", default="evaluation/baseline_latency.csv")
    parser.add_argument("--finetuned-latency", default="evaluation/finetuned_latency.csv")
    parser.add_argument("--sentences", default="evaluation/test_sentences.csv")
    parser.add_argument("--results", default="evaluation/results.csv")
    parser.add_argument("--summarize-latency", action="store_true")
    parser.add_argument("--build-scoring", action="store_true")
    parser.add_argument("--summarize-scores", action="store_true")
    args = parser.parse_args()

    if args.summarize_latency:
        summarize_latency(args.baseline_latency, args.finetuned_latency)
    if args.build_scoring:
        build_scoring_template(args.sentences, args.results)
    if args.summarize_scores:
        summarize_scores(args.results)

    if not (args.summarize_latency or args.build_scoring or args.summarize_scores):
        parser.print_help()


if __name__ == "__main__":
    main()
