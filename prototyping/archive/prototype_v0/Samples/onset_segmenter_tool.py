#!/usr/bin/env python3
import os
import argparse
import librosa
import numpy as np
import soundfile as sf

def segment_onsets_to_clips(
    input_path: str,
    output_dir: str,
    duration: float = 0.5,
    hop_length: int = 512,
    padding_ms: float = 0.0,
    backtrack: bool = False,
):
    """
    Detect onsets in `input_path`, then extract fixed-length clips
    of `duration` seconds around each onset (padding if needed), and
    save them to `output_dir`.
    """
    # 1) Load at most `duration`+padding seconds (optional optimization)
    y, sr = librosa.load(input_path, sr=None, mono=True)
    fixed_len = int(duration * sr)

    # 2) Onset detection
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr,
        hop_length=hop_length,
        backtrack=backtrack
    )
    onset_samples = librosa.frames_to_samples(onset_frames, hop_length=hop_length)

    # 3) Compute padding in samples
    pad = 0 #int(padding_ms * sr / 1000)    # PADDING OFF - won't align with dataset of set duration

    # 4) Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 5) Extract and save clips
    for i, onset in enumerate(onset_samples, start=1):
        start = onset - pad
        end   = onset + fixed_len + pad

        # Trim to audio bounds
        if start < 0:
            start = 0
        if end > len(y):
            end = len(y)

        clip = y[start:end]
        # If clip shorter than fixed_len+2*pad, pad with zeros
        target_len = fixed_len + 2 * pad
        if len(clip) < target_len:
            clip = np.pad(clip, (0, target_len - len(clip)), mode="constant")

        # Finally, to guarantee fixed_len, center-crop off the padding extras
        # (so padding only provides context, not extra length)
        clip = clip[pad : pad + fixed_len]

        # Save
        out_path = os.path.join(output_dir, f"segment_{i:03d}.wav")
        sf.write(out_path, clip, sr)
        t0 = start / sr
        print(f"{i:03d}: {t0:.3f}s → {out_path}")

def main():
    p = argparse.ArgumentParser(
        description="Onset-based segmenter: cut fixed-duration clips around each detected onset."
    )
    p.add_argument("--input",       "-i", required=True, help="Path to input .wav file")
    p.add_argument("--output_dir",  "-o", required=True, help="Directory to save clips")
    p.add_argument("--duration",    "-d", type=float, default=1.0,
                   help="Duration (in seconds) of each clip after onset")
    p.add_argument("--hop_length",  "-hl", type=int, default=512,
                   help="Hop length (samples) for onset detection")
    p.add_argument("--padding_ms",  "-p", type=float, default=0.0,
                   help="Context padding before each onset (milliseconds)")
    p.add_argument("--backtrack",   "-b", action="store_true",
                   help="Refine onsets to nearest preceding minimum")
    args = p.parse_args()

    segment_onsets_to_clips(
        input_path   = args.input,
        output_dir   = args.output_dir,
        duration     = args.duration,
        hop_length   = args.hop_length,
        padding_ms   = args.padding_ms,
        backtrack    = args.backtrack,
    )

if __name__ == "__main__":
    main()