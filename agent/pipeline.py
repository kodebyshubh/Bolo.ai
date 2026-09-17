"""
Full voice agent pipeline: audio in -> STT -> LLM (+ tools) -> TTS -> audio
out, with per-stage latency printed to the console.

Needs a GPU (for TTS) and GROQ_API_KEY set (for STT + LLM).

Usage:
    python agent/pipeline.py path/to/input.wav --out reply.wav
"""
import argparse
import os
import queue
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    # avoid UnicodeEncodeError when printing Hindi/Devanagari text on a
    # console whose default encoding isn't UTF-8 (seen on Windows and in
    # some server/subprocess contexts)
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

import soundfile as sf
from dotenv import load_dotenv

load_dotenv()

from agent.llm import respond, respond_stream
from agent.sentence_splitter import split_sentences
from agent.stt import transcribe
from agent.tts import SAMPLE_RATE, synthesize


def run(audio_input):
    """audio_input: path to an input WAV file, or raw audio bytes.
    Returns (output_audio: np.ndarray, timings: dict)."""
    timings = {}
    start_total = time.perf_counter()

    start = time.perf_counter()
    stt_result = transcribe(audio_input)
    timings["stt_seconds"] = time.perf_counter() - start
    text = stt_result["text"]
    print(f"[STT] ({stt_result['language']}): {text}")

    start = time.perf_counter()
    reply_text = respond(text)
    timings["llm_seconds"] = time.perf_counter() - start
    print(f"[LLM]: {reply_text}")

    start = time.perf_counter()
    output_audio = synthesize(reply_text)
    timings["tts_seconds"] = time.perf_counter() - start

    # No streaming synthesis yet (stretch goal, not implemented) -- the full
    # output is only available once TTS finishes, so time-to-first-audio
    # equals total latency rather than a true first-chunk measurement.
    timings["total_seconds"] = time.perf_counter() - start_total
    timings["time_to_first_audio_seconds"] = timings["total_seconds"]

    return output_audio, timings


_QUEUE_DONE = object()


def run_streaming(audio_input):
    """PRD v2 step 3+5: same STT -> LLM -> TTS pipeline as run(), but the
    LLM's reply streams and is split into sentences (agent.sentence_splitter),
    each synthesized and yielded as soon as it's ready, instead of waiting
    for the whole reply before any audio exists.

    TTS runs on a single dedicated background thread consuming a queue,
    fed by a second thread that just keeps pulling sentences from the LLM
    stream as fast as they arrive. This means: while this generator's
    caller is off doing whatever it does with sentence N's audio (playing
    it, writing it, whatever takes real wall-clock time), the background
    thread is already synthesizing sentence N+1 -- overlapping sentence
    N+1's *generation* with sentence N's *consumption*, not with sentence
    N's own generation. Exactly one thread ever calls
    agent.tts.synthesize(): a concurrency spike
    (agent/tts_concurrency_spike.py) found that two threads calling it on
    the same model instance can crash (AttributeError on Unsloth's patched
    attention layer's internal scratch state), not just fail to speed
    anything up -- so true parallel *generation* isn't attempted here.

    STT is still a single blocking call -- agent/stt.py hasn't been
    rewritten around AssemblyAI's live-upload API yet (a later step); only
    the LLM->TTS half of the pipeline streams so far.

    Yields dicts as events happen:
      {"type": "sentence_audio", "index": int, "sentence": str, "audio": np.ndarray}
      {"type": "done", "reply_text": str, "timings": dict}
    time_to_first_audio_seconds here is a real measurement (time to
    sentence 1's audio), unlike run()'s stand-in value. Each per-sentence
    entry's queue_wait_seconds is the evidence for whether overlap actually
    happened: sentence 0 always waits close to its own tts_seconds (nothing
    could be precomputed before the first sentence existed), but a later
    sentence with queue_wait_seconds much smaller than its tts_seconds
    means the background thread had already been working on it while the
    caller was still consuming an earlier sentence."""
    start_total = time.perf_counter()

    start = time.perf_counter()
    stt_result = transcribe(audio_input)
    stt_seconds = time.perf_counter() - start
    text = stt_result["text"]
    print(f"[STT] ({stt_result['language']}): {text}")

    sentence_queue = queue.Queue()
    audio_queue = queue.Queue()

    def produce_sentences():
        for i, sentence in enumerate(split_sentences(respond_stream(text))):
            sentence_queue.put((i, sentence))
        sentence_queue.put(_QUEUE_DONE)

    def synthesize_worker():
        while True:
            item = sentence_queue.get()
            if item is _QUEUE_DONE:
                audio_queue.put(_QUEUE_DONE)
                return
            index, sentence = item
            tts_start = time.perf_counter()
            audio = synthesize(sentence)
            tts_seconds = time.perf_counter() - tts_start
            audio_queue.put((index, sentence, audio, tts_seconds))

    threading.Thread(target=produce_sentences, daemon=True).start()
    threading.Thread(target=synthesize_worker, daemon=True).start()

    first_audio_time = None
    per_sentence = []
    reply_parts = []

    while True:
        wait_start = time.perf_counter()
        item = audio_queue.get()
        queue_wait_seconds = time.perf_counter() - wait_start
        if item is _QUEUE_DONE:
            break
        index, sentence, audio, tts_seconds = item
        reply_parts.append(sentence)

        if first_audio_time is None:
            first_audio_time = time.perf_counter() - start_total

        print(f"[sentence {index}] tts={tts_seconds:.2f}s queue_wait={queue_wait_seconds:.2f}s: {sentence}")
        per_sentence.append({"sentence": sentence, "tts_seconds": tts_seconds, "queue_wait_seconds": queue_wait_seconds})
        yield {"type": "sentence_audio", "index": index, "sentence": sentence, "audio": audio}

    timings = {
        "stt_seconds": stt_seconds,
        "total_seconds": time.perf_counter() - start_total,
        "time_to_first_audio_seconds": first_audio_time if first_audio_time is not None else time.perf_counter() - start_total,
        "per_sentence": per_sentence,
    }
    yield {"type": "done", "reply_text": " ".join(reply_parts), "timings": timings}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("audio_in", help="path to input WAV file")
    parser.add_argument("--out", default="pipeline_output.wav", help="path to write the output WAV (batch mode)")
    parser.add_argument("--streaming", action="store_true", help="use the sentence-streaming pipeline instead of the batch one")
    parser.add_argument("--simulate-playback", action="store_true", help="sleep for each sentence's own audio duration after writing it, standing in for a real speaker (none is wired up yet) so queue_wait_seconds actually shows the overlap benefit instead of trivially near-zero consumption time")
    args = parser.parse_args()

    if not args.streaming:
        output_audio, timings = run(args.audio_in)
        sf.write(args.out, output_audio, SAMPLE_RATE)
        print(f"\nwrote {args.out}")
        print("latency breakdown:")
        for key, value in timings.items():
            print(f"  {key}: {value:.2f}s")
        return

    out_stem = args.out.rsplit(".", 1)[0]
    for event in run_streaming(args.audio_in):
        if event["type"] == "sentence_audio":
            path = f"{out_stem}_sentence{event['index']}.wav"
            sf.write(path, event["audio"], SAMPLE_RATE)
            print(f"  wrote {path}")
            if args.simulate_playback:
                playback_seconds = len(event["audio"]) / SAMPLE_RATE
                print(f"  simulating playback for {playback_seconds:.2f}s...")
                time.sleep(playback_seconds)
        elif event["type"] == "done":
            print(f"\n[full reply]: {event['reply_text']}")
            print("latency breakdown:")
            timings = event["timings"]
            for key, value in timings.items():
                if key == "per_sentence":
                    continue
                print(f"  {key}: {value:.2f}s")
            print("  per-sentence breakdown:")
            for i, s in enumerate(timings["per_sentence"]):
                print(f"    [{i}] tts={s['tts_seconds']:.2f}s queue_wait={s['queue_wait_seconds']:.2f}s: {s['sentence']}")


if __name__ == "__main__":
    main()
