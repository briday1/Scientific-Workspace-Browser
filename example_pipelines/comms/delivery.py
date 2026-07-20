"""Window selection and ranged reads for communications recordings."""

import numpy as np

from sigvue.plugin import Delivery, DeliveryContext

from ..io.sigmf import SigMFRecording
from .models import CommsWindow


class WindowedCommsDelivery(Delivery[SigMFRecording, CommsWindow]):
    def prepare(self, recording: SigMFRecording, ui: DeliveryContext) -> CommsWindow:
        overview = ui.once(
            f"comms-power-overview:{recording.metadata_path}",
            lambda: _power_overview(recording),
        )
        start_seconds, end_seconds = ui.windowed(
            duration=recording.duration_seconds,
            default_window=min(0.012, recording.duration_seconds),
            minimum_window=min(0.002, recording.duration_seconds),
            step=min(0.001, recording.duration_seconds),
            overview=overview,
            overview_label="Mean received power (dBFS)",
            time_unit="ms",
        )
        start_sample = round(start_seconds * recording.sample_rate)
        sample_count = max(1, round((end_seconds - start_seconds) * recording.sample_rate))
        return CommsWindow(recording, start_sample, recording.read(start_sample, sample_count))


def _power_overview(recording: SigMFRecording) -> np.ndarray:
    samples = recording.read(0, recording.sample_count)
    block_count = min(240, samples.size)
    block_size = max(1, samples.size // block_count)
    blocks = samples[: block_count * block_size].reshape(block_count, block_size)
    return 10 * np.log10(np.maximum(np.mean(np.abs(blocks) ** 2, axis=1), 1e-12))
