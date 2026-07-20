#!/usr/bin/env python3
"""Generate QPSK, 16-QAM, and 64-QAM SigMF recordings."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np


SAMPLE_RATE = 800_000.0
SAMPLES_PER_SYMBOL = 8
SYMBOL_COUNT = 4_096


def constellation(order: int) -> np.ndarray:
    if order == 4:
        points = np.asarray([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j], dtype=np.complex64)
    else:
        side = int(np.sqrt(order))
        levels = np.arange(-(side - 1), side, 2, dtype=np.float32)
        points = (levels[:, None] + 1j * levels[None, :]).reshape(-1).astype(np.complex64)
    return points / np.sqrt(np.mean(np.abs(points) ** 2))


def shaped_signal(order: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    points = constellation(order)
    symbols = points[rng.integers(0, points.size, SYMBOL_COUNT)]
    impulses = np.zeros(SYMBOL_COUNT * SAMPLES_PER_SYMBOL, dtype=np.complex64)
    impulses[SAMPLES_PER_SYMBOL // 2 :: SAMPLES_PER_SYMBOL] = symbols
    pulse_time = np.arange(-4 * SAMPLES_PER_SYMBOL, 4 * SAMPLES_PER_SYMBOL + 1) / SAMPLES_PER_SYMBOL
    pulse = np.sinc(pulse_time) * np.hamming(pulse_time.size)
    pulse /= np.max(np.abs(pulse))
    samples = np.convolve(impulses, pulse, mode="same")
    noise = (rng.normal(size=samples.size) + 1j * rng.normal(size=samples.size)) * 0.025
    samples = (0.68 * samples + noise).astype(np.complex64)
    return samples / max(1.0, float(np.max(np.abs(samples))) / 0.92)


def write_recording(root: Path, modulation: str, order: int, seed: int) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    stem = modulation.lower().replace("-", "")
    metadata_path = root / f"synthetic-{stem}.sigmf-meta"
    data_path = root / f"synthetic-{stem}.sigmf-data"
    samples = shaped_signal(order, seed)
    iq = np.column_stack((samples.real, samples.imag))
    np.round(np.clip(iq, -0.999, 0.999) * 32767).astype("<i2").tofile(data_path)
    metadata = {
        "global": {
            "core:datatype": "ci16_le",
            "core:num_channels": 1,
            "core:sample_rate": SAMPLE_RATE,
            "core:version": "1.2.0",
            "core:description": f"Synthetic {modulation}",
            "core:author": "Sigvue example generator",
            "example:modulation": modulation,
            "example:samples_per_symbol": SAMPLES_PER_SYMBOL,
            "example:sample_offset": SAMPLES_PER_SYMBOL // 2,
        },
        "captures": [{
            "core:sample_start": 0,
            "core:frequency": 0.0,
            "core:datetime": datetime.now(timezone.utc).isoformat(),
        }],
        "annotations": [],
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata_path, data_path


def generate(root: Path) -> tuple[tuple[Path, Path], ...]:
    return (
        write_recording(root, "QPSK", 4, 20260721),
        write_recording(root, "16-QAM", 16, 20260722),
        write_recording(root, "64-QAM", 64, 20260723),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data/comms",
        help="Output data directory (default: example_pipelines/data/comms)",
    )
    args = parser.parse_args()
    for metadata_path, data_path in generate(args.output.resolve()):
        print(f"Wrote {metadata_path}")
        print(f"Wrote {data_path}")


if __name__ == "__main__":
    main()
