"""Window-selection policy; analysis only sees the samples delivered here."""

import numpy as np

from sigvue.plugin import Delivery, DeliveryContext

from ..io.sigmf import SigMFRecording
from .models import SignalWindow


class WindowedSamples(Delivery[SigMFRecording, SignalWindow]):
    def prepare(self, recording: SigMFRecording, ui: DeliveryContext) -> SignalWindow:
        overview = ui.once(
            f"lte-power:{recording.metadata_path}",
            lambda: power_overview(recording),
        )
        start_seconds, end_seconds = ui.windowed(
            duration=recording.duration_seconds,
            default_window=min(0.012, recording.duration_seconds),
            minimum_window=min(0.004, recording.duration_seconds),
            step=min(0.002, recording.duration_seconds),
            overview=overview,
            overview_label="Mean received power (dBFS)",
            time_unit="ms",
        )
        start = round(start_seconds * recording.sample_rate)
        count = max(1, round((end_seconds - start_seconds) * recording.sample_rate))
        return SignalWindow(recording, start, recording.read(start, count))


def power_overview(recording: SigMFRecording, bins: int = 300) -> np.ndarray:
    edges = np.linspace(0, recording.sample_count, min(bins, recording.sample_count) + 1, dtype=np.int64)
    values = []
    for start, stop in zip(edges[:-1], edges[1:]):
        samples = recording.read(int(start), int(stop - start))
        values.append(10 * np.log10(max(float(np.mean(np.abs(samples) ** 2)), 1e-12)))
    return np.asarray(values)
