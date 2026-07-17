"""Generate ignored local ci16_le data for the 10 MHz LFM collection example."""
from __future__ import annotations

import argparse
import json
from math import pi
from pathlib import Path

import numpy as np


R_OHMS = 50.0
THERMAL_NOISE_DBM_HZ = -174.0
DEFAULT_NOISE_FIGURE_DB = 7.0
VOLTS_PER_COUNT = 1e-6
OTA_PRF_HZ = 1_000.0
OTA_PULSE_WIDTH_SECONDS = 50e-6
OTA_SWEEP_BANDWIDTH_HZ = 4_000_000.0


def noise_component_std(sample_rate: float, noise_figure_db: float) -> float:
    """Return each I/Q component's voltage standard deviation for complex noise."""
    noise_psd_watts_hz = 1e-3 * 10 ** ((THERMAL_NOISE_DBM_HZ + noise_figure_db) / 10)
    noise_power_watts = noise_psd_watts_hz * sample_rate
    return np.sqrt(noise_power_watts * R_OHMS)


def write_member(
    root: Path,
    role: str,
    channel: int,
    duration: float,
    sample_rate: int,
    calibration_dbm: float,
    noise_figure_db: float,
    seed: int,
) -> None:
    count = round(duration * sample_rate)
    metadata = root / f"{role}-ch{channel}.sigmf-meta"
    data = root / f"{role}-ch{channel}.sigmf-data"
    metadata.write_text(
        json.dumps(
            {
                "global": {
                    "core:datatype": "ci16_le",
                    "core:sample_rate": sample_rate,
                    "core:num_channels": 1,
                    "core:description": f"Synthetic {role}, channel {channel}; interleaved IQ",
                }
            }
        ),
        encoding="utf-8",
    )
    amplitude = np.sqrt(2 * R_OHMS * 1e-3 * 10 ** (calibration_dbm / 10))
    noise_std = noise_component_std(sample_rate, noise_figure_db)
    generator = np.random.default_rng(seed)
    phase = (0.0, 0.37, -0.68, 1.04)[channel - 1]
    chunk = 250_000
    with data.open("wb") as stream:
        for start in range(0, count, chunk):
            index = np.arange(start, min(start + chunk, count))
            time = index / sample_rate
            if role == "calibration":
                # The calibration network supplies a pure tone at the declared
                # incident power; receiver noise is measured by separate members.
                signal = amplitude * np.exp(1j * (2 * pi * 1_000_000 * time + phase))
            else:
                noise = noise_std * (
                    generator.standard_normal(time.size) + 1j * generator.standard_normal(time.size)
                )
                if role == "terminated-noise":
                    signal = noise
                else:
                    pri_samples = round(sample_rate / OTA_PRF_HZ)
                    pulse_samples = round(sample_rate * OTA_PULSE_WIDTH_SECONDS)
                    fast_sample = np.remainder(index, pri_samples)
                    fast_time = fast_sample / sample_rate
                    pulse_active = fast_sample < pulse_samples
                    chirp_rate = OTA_SWEEP_BANDWIDTH_HZ / OTA_PULSE_WIDTH_SECONDS
                    chirp_phase = 2 * pi * (
                        -OTA_SWEEP_BANDWIDTH_HZ / 2 * fast_time
                        + 0.5 * chirp_rate * fast_time**2
                    ) + phase
                    signal = noise + pulse_active * amplitude * np.exp(1j * chirp_phase)
            iq = np.empty((len(time), 2), dtype="<i2")
            iq[..., 0] = np.clip(np.rint(signal.real / VOLTS_PER_COUNT), -32768, 32767)
            iq[..., 1] = np.clip(np.rint(signal.imag / VOLTS_PER_COUNT), -32768, 32767)
            iq.tofile(stream)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("data/lfm-collection"))
    parser.add_argument("--noise-figure-db", type=float, default=DEFAULT_NOISE_FIGURE_DB)
    parser.add_argument("--seed", type=int, default=20260717)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for path in root.glob("*.sigmf-*"):
        path.unlink()
    sample_rate, power = 10_000_000, -20.0
    members = []
    for role_index, (role, duration) in enumerate(
        (("calibration", 0.1), ("terminated-noise", 0.1), ("ota", 1.0))
    ):
        for channel in range(1, 5):
            write_member(
                root,
                role,
                channel,
                duration,
                sample_rate,
                power,
                args.noise_figure_db,
                args.seed + role_index * 100 + channel,
            )
            members.append(
                {
                    "role": role,
                    "channel": channel,
                    "metadata": f"{role}-ch{channel}.sigmf-meta",
                    "data": f"{role}-ch{channel}.sigmf-data",
                    "duration_seconds": duration,
                }
            )
    manifest = {
        "collection": {
            "name": "Synthetic 10 MHz four-channel LFM collection",
            "sample_rate": sample_rate,
            "calibration_dbm": power,
            "ota_prf_hz": OTA_PRF_HZ,
            "ota_pulse_width_seconds": OTA_PULSE_WIDTH_SECONDS,
        },
        "members": members,
    }
    (root / "lfm-10mhz.sigmf-collection").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"Generated {len(members)} members in {root} "
        f"with hidden true noise figure {args.noise_figure_db:g} dB"
    )


if __name__ == "__main__":
    main()
