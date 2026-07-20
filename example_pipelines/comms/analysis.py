"""Recover symbol samples and eye traces from one complete recording."""

import numpy as np

from sigvue.plugin import Analysis

from ..io.sigmf import SigMFRecording
from .models import CommsProducts


def process(recording: SigMFRecording, settings: None) -> CommsProducts:
    samples = recording.read(0, recording.sample_count)
    metadata = recording.metadata["global"]
    modulation = str(metadata.get("example:modulation", "Unknown"))
    samples_per_symbol = int(metadata.get("example:samples_per_symbol", 8))
    sample_offset = int(metadata.get("example:sample_offset", samples_per_symbol // 2))

    symbols = samples[sample_offset::samples_per_symbol]
    guard = 12
    symbols = symbols[guard:-guard] if symbols.size > guard * 2 else symbols

    eye_length = samples_per_symbol * 2
    eye_starts = np.arange(sample_offset, max(sample_offset, samples.size - eye_length), samples_per_symbol)
    eye_starts = eye_starts[:180]
    eye_segments = np.asarray([samples[start : start + eye_length] for start in eye_starts])
    eye_time = np.arange(eye_length) / samples_per_symbol

    constellation_extent = _comfortable_limit(symbols, 1.2)
    eye_extent = _comfortable_limit(samples, constellation_extent)
    return CommsProducts(
        modulation,
        samples_per_symbol,
        symbols,
        eye_time,
        eye_segments,
        constellation_extent,
        eye_extent,
    )


def _comfortable_limit(values: np.ndarray, fallback: float) -> float:
    values = np.asarray(values)
    if not values.size:
        return fallback
    extent = float(np.quantile(np.maximum(np.abs(values.real), np.abs(values.imag)), 0.997))
    return max(1e-6, extent * 1.15)


class CommsAnalysis(Analysis[SigMFRecording, None, CommsProducts]):
    """Framework analysis object for complete communications recordings."""

    def process(self, recording: SigMFRecording, settings: None) -> CommsProducts:
        return process(recording, settings)
