# Dataset — bolo-ai TTS fine-tuning corpus

## Public corpus

- **Source:** `SPRINGLab/IndicTTS-English` (Hugging Face), speaker: single speaker identified via the audio filename convention as `assamesefemale` (an Assamese-native female speaker reading English — Indian-accented English, a good fit for this project). No explicit speaker column on this HF mirror; confirmed single-speaker by sampling 300 rows.
- **Source:** `SPRINGLab/IndicTTS-Hindi` (Hugging Face), speaker: `hindifullfemale`, same confirmation method.
- **License:** CC BY 4.0 — mirrors of the IIT Madras IndicTTS database.
- **Citation:** IIT Madras IndicTTS database, via `SPRINGLab/IndicTTS-English` and `SPRINGLab/IndicTTS-Hindi` on Hugging Face.
- **Duration used:** English: 90.1 min (1,352 clips), Hindi: 87.5 min (652 clips, after removing 7 outlier clips over 20s).
- **Hindi text note:** Native transcripts are Devanagari script, which fragments into meaningless byte-level tokens on the `orpheus-3b-0.1-ft` tokenizer (measured: 2.08 chars/token). Transliterated to lowercased ITRANS (ASCII Latin script) before use — measured 2.71 chars/token, a real improvement. See `zdnd/decision.md` for the full investigation.

## Self-recorded supplement

- **Environment:** `<fill in — room / conditions>`
- **Mic:** `<fill in — mic model>`
- **Sampling rate at capture:** `<fill in>` (converted to 24kHz mono via `data/validate_audio.py`)
- **Duration:** 5.9 minutes, 100 clips (shorter than the original 10-15 min target — each clip is a short discrete sentence with minimal padding, not one continuous recording, so total captured speech came out lower for the same sentence count).
- **Language mix:** All 100 clips are Hinglish (code-switched English/Hindi), covering greetings, order-status domain phrases, numbers/dates, Indian names, emotion-tagged lines, and general conversation. See `data/hinglish_recording_script.txt` for the full script.

## Combined dataset

- **Total duration:** 3.06 hours (183.5 min: 90.1 English + 87.5 Hindi + 5.9 self-recorded Hinglish).
- **Format:** WAV, 24kHz, mono.
- **Rows:** 2,104 total (1,352 English + 652 Hindi + 100 self-recorded Hinglish).
- **Metadata:** `data/metadata.csv` (filename,text), emotion tags in `<angle_brackets>` where applicable (self-recorded rows only).

## Rejected takes

| File / take | Reason rejected |
|---|---|
| `public/hi_00386.wav` | Too long (25.25s), flagged by `validate_audio.py` |
| `public/hi_00388.wav` | Too long (23.15s) |
| `public/hi_00550.wav` | Too long (20.55s) |
| `public/hi_00573.wav` | Too long (20.99s) |
| `public/hi_00618.wav` | Too long (20.13s) |
| `public/hi_00637.wav` | Too long (20.27s) |
| `public/hi_00652.wav` | Too long (20.42s) |
