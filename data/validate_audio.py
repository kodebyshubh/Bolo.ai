"""
Scan a folder of WAV files, ensure each is mono at 24kHz (resampling/converting
in place if not), and flag files under 1s or over 20s as candidates for review.

Usage:
    python data/validate_audio.py data/audio/public
    python data/validate_audio.py data/audio/selfrecorded
"""
import argparse
import glob
import os

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 24000
MIN_SECONDS = 1.0
MAX_SECONDS = 20.0


def validate_file(path):
    data, sr = sf.read(path, always_2d=False)
    changed = False

    if data.ndim > 1:
        data = np.mean(data, axis=1)
        changed = True

    if sr != TARGET_SR:
        data = librosa.resample(np.asarray(data, dtype=np.float32), orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR
        changed = True

    if changed:
        sf.write(path, data, sr, subtype="PCM_16")

    duration = len(data) / sr
    flag = None
    if duration < MIN_SECONDS:
        flag = f"too short ({duration:.2f}s)"
    elif duration > MAX_SECONDS:
        flag = f"too long ({duration:.2f}s)"

    return changed, duration, flag


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", help="folder of WAV files to validate")
    args = parser.parse_args()

    wav_paths = sorted(glob.glob(os.path.join(args.folder, "*.wav")))
    if not wav_paths:
        print(f"no .wav files found in {args.folder}")
        return

    fixed = []
    flagged = []
    for path in wav_paths:
        changed, duration, flag = validate_file(path)
        if changed:
            fixed.append(path)
        if flag:
            flagged.append((path, flag))

    print(f"checked {len(wav_paths)} files in {args.folder}")
    print(f"resampled/converted to {TARGET_SR}Hz mono: {len(fixed)}")
    for path in fixed:
        print(f"  fixed: {path}")

    print(f"flagged for review (outside {MIN_SECONDS}-{MAX_SECONDS}s): {len(flagged)}")
    for path, flag in flagged:
        print(f"  {path}: {flag}")


if __name__ == "__main__":
    main()
