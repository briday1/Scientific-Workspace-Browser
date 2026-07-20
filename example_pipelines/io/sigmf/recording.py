"""Framework-independent SigMF metadata and ranged sample access."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from sigvue.plugin import DataResource


@dataclass(frozen=True)
class SigMFRecording:
    metadata_path: Path
    data_path: Path
    sample_rate: float
    sample_count: int
    center_frequency: float
    metadata: dict[str, object]

    @property
    def duration_seconds(self) -> float:
        return self.sample_count / self.sample_rate

    def read(self, start: int, count: int) -> np.ndarray:
        """Read only the requested complex frames from interleaved int16 data."""
        start = min(self.sample_count, max(0, int(start)))
        count = min(max(0, int(count)), self.sample_count - start)
        with self.data_path.open("rb") as stream:
            stream.seek(start * 4)
            scalars = np.fromfile(stream, dtype="<i2", count=count * 2)
        frames = scalars.reshape(-1, 2).astype(np.float32)
        return np.asarray((frames[:, 0] + 1j * frames[:, 1]) / 32768.0, dtype=np.complex64)


def load_metadata(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_recording(path: Path) -> SigMFRecording:
    metadata = load_metadata(path)
    global_metadata = metadata["global"]
    if global_metadata.get("core:datatype") != "ci16_le":
        raise ValueError("The LTE example expects one-channel ci16_le SigMF data")
    if int(global_metadata.get("core:num_channels", 1)) != 1:
        raise ValueError("The LTE example expects one channel per SigMF recording")
    sample_rate = float(global_metadata["core:sample_rate"])
    captures = metadata.get("captures", ())
    center_frequency = float(captures[0].get("core:frequency", 0.0)) if captures else 0.0
    data_path = path.with_name(path.name.removesuffix(".sigmf-meta") + ".sigmf-data")
    return SigMFRecording(
        path,
        data_path,
        sample_rate,
        data_path.stat().st_size // 4,
        center_frequency,
        metadata,
    )


def describe_recording(path: Path) -> DataResource:
    metadata = load_metadata(path)
    global_metadata = metadata["global"]
    captures = metadata.get("captures", ())
    capture = captures[0] if captures else {}
    return DataResource(
        identifier=path.name.removesuffix(".sigmf-meta"),
        title=str(global_metadata.get("core:description") or path.stem),
        source=path,
        subtitle=f"{float(global_metadata['core:sample_rate']) / 1e6:g} MS/s · synthetic ci16_le",
        timestamp=datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc),
        tags=("sigmf", "synthetic", str(global_metadata.get("example:direction", "LTE"))),
        summary={
            "date": capture.get("core:datetime") or global_metadata.get("core:datetime"),
            "sample_rate": global_metadata.get("core:sample_rate"),
            "rf_frequency": capture.get("core:frequency"),
        },
        navigation_path=(path.parent.name,),
    )
