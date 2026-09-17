"""
Generate audio from unsloth/orpheus-3b-0.1-ft for every sentence in
evaluation/test_sentences.csv. Without --adapter, this is the "before
fine-tuning" baseline; with --adapter pointing at a saved LoRA checkpoint
(e.g. training/checkpoints/final_adapter), this generates the "after"
side of Phase 4's base-vs-finetuned comparison.

Needs a GPU (Colab T4 or better) -- unsloth + torch, plus the SNAC 24kHz
vocoder that decodes Orpheus's discrete audio tokens into a waveform.

Control-token IDs below (START_OF_HUMAN etc.) were verified against the
actual unsloth/orpheus-3b-0.1-ft tokenizer: this tokenizer exposes them as
<custom_token_N> entries in get_added_vocab(), not as the human-readable
"<|start_of_human|>"-style strings some Orpheus docs use -- those resolve to
None via convert_tokens_to_ids on this checkpoint. Confirmed end-to-end with
a single-sentence generation that decoded to clear, audible speech.

Usage:
    python inference/tts_server.py                                          # baseline
    python inference/tts_server.py --adapter training/checkpoints/final_adapter  # fine-tuned
"""
import argparse
import csv
import os
import time

import soundfile as sf
import torch
from snac import SNAC
from unsloth import FastLanguageModel

MODEL_NAME = "unsloth/orpheus-3b-0.1-ft"
SNAC_MODEL_NAME = "hubertsiuzdak/snac_24khz"
SAMPLE_RATE = 24000

# Orpheus control token IDs (verified against the loaded tokenizer's added vocab)
START_OF_SPEECH = 128257
END_OF_SPEECH = 128258
START_OF_HUMAN = 128259
END_OF_HUMAN = 128260
START_OF_AI = 128261
AUDIO_TOKEN_OFFSET = 128266  # first audio-code token; maps Orpheus's audio codes into the LLM vocab
TEXT_EOT = 128009  # Llama-3 <|eot_id|>, closes the human turn before END_OF_HUMAN


def load_models(adapter_path=None):
    """adapter_path: directory saved via model.save_pretrained() after LoRA
    fine-tuning (contains adapter_config.json + adapter_model.safetensors).
    Unsloth loads base weights + adapter together when given that directory
    as model_name, since adapter_config.json records the base model name."""
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=adapter_path or MODEL_NAME,
        max_seq_length=2048,
        dtype=torch.bfloat16,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)

    snac_model = SNAC.from_pretrained(SNAC_MODEL_NAME).eval()
    if torch.cuda.is_available():
        snac_model = snac_model.to("cuda")

    return model, tokenizer, snac_model


def build_prompt(tokenizer, text, voice=None):
    prompt_text = f"{voice}: {text}" if voice else text
    text_ids = tokenizer(prompt_text, return_tensors="pt").input_ids
    prefix = torch.tensor([[START_OF_HUMAN]], dtype=torch.long)
    suffix = torch.tensor([[TEXT_EOT, END_OF_HUMAN, START_OF_AI, START_OF_SPEECH]], dtype=torch.long)
    return torch.cat([prefix, text_ids, suffix], dim=1)


def tokens_to_audio(snac_model, token_ids):
    """Orpheus emits 7 audio codes per frame across 3 hierarchical SNAC layers."""
    codes = [t - AUDIO_TOKEN_OFFSET for t in token_ids if t >= AUDIO_TOKEN_OFFSET]
    n_frames = len(codes) // 7
    codes = codes[: n_frames * 7]
    if n_frames == 0:
        return None

    layer1, layer2, layer3 = [], [], []
    for i in range(n_frames):
        frame = codes[i * 7:(i + 1) * 7]
        layer1.append(frame[0])
        layer2.append(frame[1] - 4096)
        layer2.append(frame[4] - 4096 * 4)
        layer3.append(frame[2] - 4096 * 2)
        layer3.append(frame[3] - 4096 * 3)
        layer3.append(frame[5] - 4096 * 5)
        layer3.append(frame[6] - 4096 * 6)

    device = next(snac_model.parameters()).device
    codes_tensor = [
        torch.tensor(layer1, device=device).unsqueeze(0),
        torch.tensor(layer2, device=device).unsqueeze(0),
        torch.tensor(layer3, device=device).unsqueeze(0),
    ]
    with torch.no_grad():
        audio = snac_model.decode(codes_tensor)
    return audio.squeeze().cpu().numpy()


def synthesize(model, tokenizer, snac_model, text, voice=None):
    input_ids = build_prompt(tokenizer, text, voice=voice).to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=1200,
            do_sample=False,
            repetition_penalty=1.3,
            eos_token_id=END_OF_SPEECH,
        )
    generated = output_ids[0, input_ids.shape[1]:].tolist()
    if END_OF_SPEECH in generated:
        # generate()'s eos_token_id doesn't reliably stop sampling on this
        # tokenizer's added vocab -- it can run past END_OF_SPEECH into a
        # second/third repeat of the same utterance. Truncate to the first
        # speech segment ourselves.
        generated = generated[: generated.index(END_OF_SPEECH)]
    else:
        print(f"warning: no END_OF_SPEECH in {len(generated)} generated tokens for {text!r} -- output may be truncated mid-utterance or contain trailing noise")
    return tokens_to_audio(snac_model, generated)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sentences", default="evaluation/test_sentences.csv")
    parser.add_argument("--adapter", default=None, help="path to a saved LoRA adapter dir; omit for the base-model baseline")
    parser.add_argument("--outdir", default=None, help="default: evaluation/baseline_outputs, or finetuned_outputs if --adapter is given")
    parser.add_argument("--latency-out", default=None, help="default: evaluation/baseline_latency.csv, or finetuned_latency.csv if --adapter is given")
    parser.add_argument("--voice", default=None, help="optional Orpheus voice tag prefix")
    args = parser.parse_args()

    label = "finetuned" if args.adapter else "baseline"
    outdir = args.outdir or f"evaluation/{label}_outputs"
    latency_out = args.latency_out or f"evaluation/{label}_latency.csv"

    os.makedirs(outdir, exist_ok=True)
    model, tokenizer, snac_model = load_models(adapter_path=args.adapter)

    with open(args.sentences, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    latency_rows = []
    for row in rows:
        sentence_id = row["id"]
        text = row["sentence"]
        out_path = os.path.join(outdir, f"{sentence_id}.wav")

        start = time.perf_counter()
        audio = synthesize(model, tokenizer, snac_model, text, voice=args.voice)
        elapsed = time.perf_counter() - start

        if audio is None:
            print(f"warning: {sentence_id}: no audio tokens generated, skipping")
            continue

        sf.write(out_path, audio, SAMPLE_RATE)
        print(f"{sentence_id}: {elapsed:.2f}s -> {out_path}")
        latency_rows.append({"id": sentence_id, "language": row["language"], "generation_seconds": f"{elapsed:.3f}"})

    with open(latency_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "language", "generation_seconds"])
        writer.writeheader()
        writer.writerows(latency_rows)
    print(f"\nwrote {len(latency_rows)} latency rows to {latency_out}")


if __name__ == "__main__":
    main()
