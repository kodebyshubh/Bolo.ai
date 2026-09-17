"""
PRD v2 step 4 spike (not production code): does calling agent.tts.synthesize()
from two threads at once actually overlap on the GPU, or does it just
serialize? Determines what "async parallel synthesis" can realistically mean
before building the real implementation, per zdnd/action_plan.md.

Needs a GPU -- run on Kaggle/Colab.

Usage:
    python agent/tts_concurrency_spike.py
"""
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agent.tts import synthesize

SENTENCES = [
    "This is the first test sentence for the concurrency spike.",
    "This is the second test sentence for the concurrency spike.",
]


def main():
    # load the model once, up front, so neither timed section below
    # includes model-load cost
    print("warming up (loads the model)...")
    synthesize("warm up")
    print("warmed up.\n")

    print("=== sequential baseline ===")
    start = time.perf_counter()
    for s in SENTENCES:
        synthesize(s)
    sequential_seconds = time.perf_counter() - start
    print(f"sequential total: {sequential_seconds:.2f}s\n")

    print("=== two threads calling synthesize() at once ===")
    errors = {}
    finish_times = {}

    def worker(index, text, start_time):
        try:
            synthesize(text)
            finish_times[index] = time.perf_counter() - start_time
        except Exception as e:
            errors[index] = repr(e)

    start = time.perf_counter()
    threads = [threading.Thread(target=worker, args=(i, s, start)) for i, s in enumerate(SENTENCES)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    threaded_seconds = time.perf_counter() - start

    print(f"threaded total: {threaded_seconds:.2f}s")
    print(f"per-thread finish times: {finish_times}")
    print(f"errors: {errors}")
    print()

    print("=== verdict ===")
    if errors:
        print("at least one thread errored -- synthesize() is not safely callable from two threads on the same model instance concurrently. Do not attempt naive threading; look into request queuing or separate model instances per GPU instead.")
    elif threaded_seconds < sequential_seconds * 0.85:
        print(f"threaded ({threaded_seconds:.2f}s) meaningfully beat sequential ({sequential_seconds:.2f}s) -- the GPU/driver can usefully overlap independent generate() calls. True parallel synthesis is worth building.")
    else:
        print(f"threaded ({threaded_seconds:.2f}s) ~= sequential ({sequential_seconds:.2f}s) -- calls are effectively serializing on the GPU despite being issued from separate threads. 'Parallel synthesis' should mean overlapping sentence N+1's generation with sentence N's *playback*, not with sentence N's own generation.")


if __name__ == "__main__":
    main()
