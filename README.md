# Bolo.ai — Indian Multilingual Conversational Voice Agent

## Problem

Most portfolio voice-AI projects stop at "I called an API and it talked back." That doesn't demonstrate training/fine-tuning your own voice model, handling Indian-accent and code-switched Hindi/English speech, wiring tool calling into an agent, or evaluating quality scientifically instead of by ear.

Bolo.ai is a small but complete voice agent built to close that gap: a LoRA-fine-tuned TTS voice trained on a self-recorded + public Indian-English/Hinglish dataset, wired into a full STT → LLM (tool-calling) → TTS pipeline, with a measured baseline-vs-fine-tuned comparison to back up the improvement claim — plus a real-time streaming layer on top (sentence-level TTS streaming, LLM token streaming, WebSocket transport) built as a second phase to push per-turn latency down and demonstrate a real-time systems architecture, not just a batch script.

## Architecture

**Voice agent pipeline** (`agent/`):

```
 USER (mic)  →  STT (AssemblyAI)  →  TEXT  →  LLM (Groq, tool-calling)
                                                     │
                                    ┌────────────────┴────────────────┐
                                    ▼                                 ▼
                             Tool Call (get_order_status)        Direct Reply
                                    └────────────────┬────────────────┘
                                                      ▼
                                              Response text
                                                      │
                                                      ▼
                                    TTS (fine-tuned Orpheus-3B + SNAC vocoder)
                                                      │
                                                      ▼
                                                  🔊 AUDIO
```

**Fine-tuning sub-pipeline** (offline, produces the TTS component above):

```
Self-recorded Hinglish + public IndicTTS corpus (metadata.csv)
        │
        ▼
Unsloth + LoRA fine-tune of unsloth/orpheus-3b-0.1-ft (Kaggle T4)
        │
        ▼
Fine-tuned adapter (training/checkpoints/final_adapter)
        │
        ▼
Base vs. fine-tuned blind evaluation (evaluation/)
        │
        ▼
TTS component used by the live agent
```

**Real-time streaming layer** (added in a second build phase, on top of the pipeline above): the LLM's reply streams token-by-token, a sentence splitter (`agent/sentence_splitter.py`, handles both Latin punctuation and the Hindi danda `।`) emits complete sentences as soon as they're ready, and a single background worker thread synthesizes each sentence's audio while the caller consumes the previous one — so sentence 2's generation overlaps sentence 1's playback instead of waiting for the whole reply. `agent/server.py` exposes this over a WebSocket (`/ws/converse`): the browser (`frontend/index.html`) records a full utterance, sends it as one binary message, and receives back a JSON metadata frame + a WAV audio frame per sentence, in order, ending with a `done` frame carrying the full reply text and a per-stage latency breakdown.

There's no database — order data is an in-memory mock dict (`agent/tools.py`), and dataset/eval state lives in CSV files under `data/` and `evaluation/`. There's no user auth — this is a single-user local/demo deployment, not a multi-tenant service.

## Dataset

- **Public corpus:** `SPRINGLab/IndicTTS-English` (Assamese-native speaker reading English, Indian-accented) and `SPRINGLab/IndicTTS-Hindi`, each filtered to a single consistent speaker. 90.1 min English (1,352 clips) + 87.5 min Hindi (652 clips). CC BY 4.0, mirrors of the IIT Madras IndicTTS database.
- **Self-recorded supplement:** 5.9 minutes / 100 clips of code-switched Hinglish (greetings, order-status domain phrases, numbers, names, emotion-tagged lines) — the public corpora are single-language and don't cover mixed-language sentences at all.
- **Hindi text is transliterated, not native Devanagari.** Raw Devanagari fragments into meaningless byte-level tokens on the Llama-3-based Orpheus tokenizer (measured 2.08 chars/token); lowercased ITRANS transliteration tokenizes far more coherently (2.71 chars/token). This means "Hindi support" here is phonetic-romanization support, not native-script support — a real scope boundary, not an oversight.
- **Combined:** 3.06 hours, 2,104 rows, WAV/24kHz/mono, `data/metadata.csv`. Full sourcing, license, and rejected-takes detail in `data/README.md`.

## Fine-tuning

`unsloth/orpheus-3b-0.1-ft` fine-tuned with LoRA (16-bit, via Unsloth) on Kaggle's free T4 GPU tier, one epoch over the full 2,104-row dataset (~31 min). Training loss dropped from ~4.65 to ~4.0.

Audio is converted to Orpheus's discrete token representation manually (`audio_to_tokens`, the verified mathematical inverse of the inference-side SNAC decoder) rather than trusting an assumed Unsloth utility, since no such utility's exact API surface could be confirmed in advance for this Unsloth version. A round-trip encode→decode sanity check on a real training clip was required before any dataset preprocessing was trusted.

Generation uses **greedy decoding** (`do_sample=False`), not sampling — found necessary after sampled decoding produced non-terminating generations and mid-utterance phrase repetition on some sentences; greedy decoding converges reliably to the model's own `END_OF_SPEECH` token. Orpheus's control tokens (start/end of human turn, start of AI turn, start/end of speech) were verified against the actual loaded tokenizer's `get_added_vocab()` rather than the human-readable strings commonly documented for Orpheus, which don't resolve to real token IDs on this checkpoint.

## Results

Base vs fine-tuned comparison on the 20-sentence held-out test set (`evaluation/results.csv`), scored blind (rater did not know which clip was which while scoring, per `evaluation/score.html`):

| Metric | Baseline | Fine-tuned | Δ |
|---|---|---|---|
| Naturalness (1-5) | 2.80 | 3.50 | +0.70 |
| Similarity to reference voice (1-5) | 1.95 | 2.15 | +0.20 |
| Pronunciation correct (0-1) | 0.40 | 0.65 | +0.25 |
| Latency, mean (s) | 15.50 | 16.06 | +0.56 |
| Latency, max (s) | 38.15 | 24.18 | -13.97 |

Fine-tuning produced a clear improvement in pronunciation accuracy (0.40 → 0.65, +62% relative) and a solid gain in perceived naturalness (2.80 → 3.50), most likely because the training data was heavily weighted toward exactly the content the base model struggled with — transliterated Hindi and code-switched Hinglish — so even one epoch of LoRA touch-up meaningfully shifted the model away from its earlier gibberish/mispronunciation failures on that content. Speaker similarity to the reference voice improved only modestly (1.95 → 2.15), which makes sense given the self-recorded target-speaker audio was just 5.9 minutes out of a 3-hour training set (the rest being two other public-corpus speakers) — one epoch of light LoRA adaptation on that small a proportion of target-speaker data isn't enough to strongly pull the model's voice identity toward it. Generation latency didn't meaningfully change on average, though the fine-tuned model's worst-case latency dropped substantially, suggesting it produces fewer of the runaway-generation edge cases the base model occasionally hit.

**Real-time streaming, measured on real end-to-end runs (browser → Cloudflare tunnel → Kaggle T4):** sentence-level streaming genuinely overlaps generation with consumption — a real trial with simulated playback showed a later sentence's queue-wait time drop by almost exactly the previous sentence's playback duration, confirming the overlap mechanism works. But raw per-sentence TTS generation itself is slow on this hardware (roughly 10-60s depending on sentence length and GPU load), so time-to-first-audio in a real deployed run has measured around 60s cold (includes one-time model load) and per-turn totals of 1-3 minutes for multi-sentence replies — see Limitations.

## How to Run

**Prerequisites:** a GPU (Kaggle/Colab free tier works; TTS needs ~6GB+ VRAM), API keys for Groq (LLM) and AssemblyAI (STT), and `HF_TOKEN` (Hugging Face, avoids rate-limited model downloads).

1. Clone the repo and `pip install -r requirements.txt`.
2. Copy `.env.example` to `.env` and fill in `GROQ_API_KEY`, `ASSEMBLYAI_API_KEY`, `HF_TOKEN`.
3. Get a fine-tuned adapter: either train one (`training/` notebook, needs the dataset built per `data/README.md`) or point `agent/tts.py`'s `DEFAULT_ADAPTER_PATH` at your own checkpoint under `training/checkpoints/`.
4. **Simplest path — one-shot CLI (batch mode):**
   ```
   python agent/pipeline.py path/to/input.wav --out reply.wav
   ```
   Runs STT → LLM → TTS once and writes the spoken reply to `reply.wav`, printing per-stage latency.
5. **Real-time streaming demo (mic → live playback in browser):**
   - Start the WebSocket server (needs the GPU environment): `python -m uvicorn agent.server:app --host 0.0.0.0 --port 8000`
   - Expose it if the server and browser aren't on the same machine (e.g. a Kaggle notebook + your laptop): a free Cloudflare quick tunnel — `cloudflared tunnel --url http://localhost:8000` — needs no signup.
   - Serve `frontend/index.html` (e.g. `python -m http.server` from the `frontend/` folder) and open it in a browser.
   - Paste the server's WebSocket URL (`wss://.../ws/converse`) into the page's Server field, click the record button, speak, and listen to the reply play back sentence by sentence.

## Limitations

- **STT on Hinglish is inconsistent, especially proper names.** English and pure Hindi transcribe reliably, but code-switched Hinglish is hit-or-miss: straightforward sentences transcribe correctly, but proper names get mangled (e.g. "Shubh" → "शुबूल"). This persisted across two different STT providers (Groq Whisper, then AssemblyAI), and explicit language hints don't fix it — forcing a single language on mixed-language audio makes results worse, not better. This is a known limitation of general-purpose ASR on code-switched speech, not something fixable at the prompt/hint level; training a custom ASR model is out of scope for this project.
- **Sub-second time-to-first-audio is not achievable with this model on free-tier GPU hardware.** A single sentence's TTS generation alone takes roughly 10-60 seconds on a Kaggle T4, regardless of streaming/overlap architecture — the streaming work (sentence splitting, generation/playback overlap) genuinely improves the experience but cannot manufacture sub-second first-audio out of a model whose own per-sentence decode time exceeds that by 1-2 orders of magnitude. Hitting that target for real would need a different, purpose-built low-latency TTS serving stack (or paid inference infrastructure), not more architecture on top of this model.
- **Generation quality degrades on long, out-of-distribution English text.** Short, casual sentences (closer to the fine-tuning data's actual style) generate at a normal pace and pitch; a long, formal English paragraph was observed to generate audibly slowed/distorted-sounding speech in one real test. This is a fine-tuned-checkpoint characteristic on unfamiliar content, confirmed by ruling out WAV encoding (sample rate, bit depth) as the cause via direct byte-level testing — not a transport or playback bug.
- **The live server handles one conversation at a time.** `agent/server.py`'s WebSocket handler calls a blocking generator directly inside an `async def` handler, which blocks the event loop for the duration of one turn — an accepted trade-off for a single-connection portfolio demo, not something that scales to concurrent users without further work (see Future Work).
- **Speaker similarity to the target voice is modest (1.95 → 2.15).** The self-recorded target-speaker audio is only 5.9 minutes of a 3-hour training set; one epoch of light LoRA adaptation on that small a fraction isn't enough to strongly pull the model's voice identity toward it.
- **The generation/playback overlap mechanism doesn't get an organic benefit in the live deployed server**, unlike in controlled local testing. The server forwards each sentence to the client as fast as it's generated, with no signal about how long the browser actually takes to play it back — so the background TTS worker never gets a real head start from actual playback time the way a simulated-delay test harness showed it could. The mechanism itself is correct and verified; the live deployment just doesn't yet feed it the pacing information it would need to show a benefit end-to-end.

## Future Work

- A client-to-server playback acknowledgment (e.g. "finished playing sentence N") so the live server's overlap mechanism gets the real pacing signal it needs to show benefit outside of simulated tests.
- `asyncio.to_thread` (or a worker-pool) around `run_streaming()` in `agent/server.py` so the WebSocket handler stops blocking the event loop, enabling real concurrent multi-user connections.
- A larger self-recorded target-speaker dataset (the current 5.9 minutes is a small fraction of the 3-hour training set) to improve speaker-similarity scores.
- True incremental mic-chunk upload (VAD-driven, using `agent/vad.py`'s already-built end-of-speech detection) instead of the current record-full-utterance-then-upload flow, plus AssemblyAI-side partial transcripts if a future SDK version exposes them.
- Investigate whether a different TTS serving approach (quantization, a different backend, or paid low-latency inference infra) could meaningfully close the gap toward the original sub-second TTFA target — deliberately not attempted in this build given the real risk (re-verifying the entire token/decoding scheme from scratch) versus uncertain payoff.
- A visible, per-sentence transcript/audio player in the dashboard (partially done — each sentence now has a scrubbable `<audio controls>` element) plus a raw-STT-transcript display for full transparency into what the agent heard.
- Authentication and rate limiting on `/ws/converse` before any public, persistent deployment — currently there's no credential check at all, which is only safe because the Cloudflare tunnel URLs used are ephemeral and effectively private.
- A model warm-up call at server startup instead of on the first real request — the one-time ~40-45s model-load cost currently lands on whichever user happens to send the first message after the server starts.
