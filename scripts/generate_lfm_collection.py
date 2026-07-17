"""Generate ignored local ci16_le data for the 10 MHz LFM collection example."""
from __future__ import annotations

import argparse
import json
from math import pi
from pathlib import Path

import numpy as np


def write_member(root: Path, role: str, channel: int, duration: float, sample_rate: int, calibration_dbm: float) -> None:
    count = round(duration * sample_rate)
    metadata = root / f"{role}-ch{channel}.sigmf-meta"
    data = root / f"{role}-ch{channel}.sigmf-data"
    metadata.write_text(json.dumps({"global": {"core:datatype": "ci16_le", "core:sample_rate": sample_rate, "core:num_channels": 1, "core:description": f"Synthetic {role}, channel {channel}; interleaved IQ"}}), encoding="utf-8")
    amplitude = np.sqrt(2 * 50 * 1e-3 * 10 ** (calibration_dbm / 10))
    phase = (0.0, 0.37, -0.68, 1.04)[channel - 1]
    chunk = 250_000
    with data.open("wb") as stream:
        for start in range(0, count, chunk):
            index = np.arange(start, min(start + chunk, count))
            time = index / sample_rate
            if role == "calibration":
                signal = amplitude * np.exp(1j * (2 * pi * 1_000_000 * time + phase))
            elif role == "terminated-noise":
                signal = amplitude * .04 * (np.sin(2*pi*time*(137_000 + channel*700)) + 1j*np.cos(2*pi*time*(211_000 + channel*500)))
            else:
                chirp_phase = 2*pi*(-2_000_000*time + 2_000_000*time**2/duration) + phase
                signal = amplitude*np.exp(1j*chirp_phase) + amplitude*.025*(np.sin(2*pi*time*(91_000 + channel*900)) + 1j*np.cos(2*pi*time*(173_000 + channel*600)))
            iq = np.empty((len(time), 2), dtype="<i2")
            iq[..., 0] = np.clip(np.rint(signal.real / 1e-6), -32768, 32767)
            iq[..., 1] = np.clip(np.rint(signal.imag / 1e-6), -32768, 32767)
            iq.tofile(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/lfm-collection"))
    args = parser.parse_args()
    root = args.output.resolve(); root.mkdir(parents=True, exist_ok=True)
    for path in root.glob("*.sigmf-*"):
        path.unlink()
    sample_rate, power = 10_000_000, -20.0
    members = []
    for role, duration in (("calibration", .1), ("terminated-noise", .1), ("ota", 1.0)):
        for channel in range(1, 5):
            write_member(root, role, channel, duration, sample_rate, power)
            members.append({"role": role, "channel": channel, "metadata": f"{role}-ch{channel}.sigmf-meta", "data": f"{role}-ch{channel}.sigmf-data", "duration_seconds": duration})
    manifest = {"collection": {"name": "Synthetic 10 MHz four-channel LFM collection", "sample_rate": sample_rate, "calibration_dbm": power}, "members": members}
    (root / "lfm-10mhz.sigmf-collection").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
