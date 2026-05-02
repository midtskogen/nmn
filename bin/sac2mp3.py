#!/usr/bin/env python3
"""Convert a SAC seismogram to MP3 audio.

Usage:
    python sac2mp3.py INPUT.sac [OUTPUT.mp3] [--speed FACTOR]

The seismic signal is normalised to 16-bit PCM and optionally sped up
(default 100×) so that low-frequency content becomes audible.
Requires: obspy, numpy, pydub (and ffmpeg installed on the system).
"""

import argparse
import io
import sys
from pathlib import Path

import numpy as np
from obspy import read
from pydub import AudioSegment


def sac_to_mp3(sac_path: str, mp3_path: str, speed_factor: float = 100.0) -> None:
    st = read(sac_path)
    tr = st[0]

    original_sr = int(tr.stats.sampling_rate)          # e.g. 100 Hz
    target_sr = int(original_sr * speed_factor)         # e.g. 10 000 Hz

    # Normalise to signed 16-bit range
    data = tr.data.astype(np.float64)
    peak = np.max(np.abs(data))
    if peak > 0:
        data = data / peak
    samples = (data * 32767).astype(np.int16)

    # Build a WAV in memory via pydub
    audio = AudioSegment(
        samples.tobytes(),
        frame_rate=target_sr,
        sample_width=2,       # 16-bit
        channels=1,
    )

    audio.export(mp3_path, format="mp3")
    duration_s = len(audio) / 1000.0
    print(f"Wrote {mp3_path}  "
          f"(speed ×{speed_factor}, effective sample rate {target_sr} Hz, "
          f"{duration_s:.1f} s)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a SAC seismogram to audible MP3.")
    parser.add_argument("input", help="Input SAC file")
    parser.add_argument("output", nargs="?", default=None,
                        help="Output MP3 file (default: same name, .mp3)")
    parser.add_argument("--speed", type=float, default=100.0,
                        help="Speed-up factor (default 100)")
    args = parser.parse_args()

    sac_path = args.input
    if not Path(sac_path).exists():
        sys.exit(f"File not found: {sac_path}")

    mp3_path = args.output or str(Path(sac_path).with_suffix(".mp3"))
    sac_to_mp3(sac_path, mp3_path, speed_factor=args.speed)


if __name__ == "__main__":
    main()
