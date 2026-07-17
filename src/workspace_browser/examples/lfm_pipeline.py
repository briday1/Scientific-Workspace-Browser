"""Shared LFM collection loading, delivery policies, analysis, and Plotly views."""
from __future__ import annotations

from dataclasses import dataclass
import json
from math import ceil, log10, pi, sqrt
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from workspace_browser.plugin import AnalysisContext, AnalysisWorkspace, DataResource, DirectorySource

from .plot_style import ORANGE, TEAL, style_plotly


R_OHMS = 50.0
THERMAL_NOISE_DBM_HZ = -174.0


@dataclass(frozen=True)
class CollectionMember:
    role: str
    channel: int
    metadata_path: Path
    data_path: Path
    duration: float


@dataclass(frozen=True)
class LfmCollection:
    sample_rate: float
    calibration_dbm: float
    adc_bits: int
    members: dict[str, tuple[CollectionMember, ...]]

    def sample_count(self, role: str) -> int:
        return round(self.members[role][0].duration * self.sample_rate)

    def read(self, role: str, start: int = 0, count: int | None = None) -> np.ndarray:
        available = self.sample_count(role)
        count = available - start if count is None else min(count, available - start)
        channels = []
        for member in self.members[role]:
            with member.data_path.open("rb") as stream:
                stream.seek(start * 4)
                iq = np.fromfile(stream, dtype="<i2", count=count * 2).reshape(-1, 2)
            channels.append(iq[:, 0].astype(np.float32) + 1j * iq[:, 1].astype(np.float32))
        return np.asarray(channels, dtype=np.complex64)


@dataclass(frozen=True)
class LfmInput:
    sample_rate: float
    calibration_dbm: float
    adc_bits: int
    pri_samples: int
    start_sample: int
    calibration_counts: np.ndarray
    noise_counts: np.ndarray
    ota_counts: np.ndarray


class BufferedDelivery:
    """Framework policy for playback: deliver one requested OTA window."""

    def prepare(self, collection: LfmCollection, ui: AnalysisContext) -> LfmInput:
        buffer_seconds = ui.number("buffer_seconds", default=0.002, minimum=0.0001, maximum=0.1, step=0.0001)
        pri_seconds = ui.number("pri_seconds", default=0.00002, minimum=0.00001, maximum=0.001, step=0.00001)
        seek_seconds = ui.number("seek_seconds", default=0.01, minimum=0.001, step=0.001)
        refresh_seconds = ui.number("refresh_seconds", default=0.15, minimum=0.05, step=0.05)
        size = min(collection.sample_count("ota"), max(1, round(buffer_seconds * collection.sample_rate)))
        pri = min(size, max(8, round(pri_seconds * collection.sample_rate)))
        time = ui.playback(
            duration=max(0.0, collection.members["ota"][0].duration - size / collection.sample_rate),
            step=seek_seconds,
            refresh_interval=refresh_seconds,
        )
        start = min(round(time * collection.sample_rate), collection.sample_count("ota") - size)
        return _input(collection, start=start, count=size, pri=pri, ui=ui)


class WholeFileDelivery:
    """Framework policy for batch mode: deliver the complete OTA member files."""

    def __init__(self, *, pri_seconds: float = 0.00002) -> None:
        self.pri_seconds = pri_seconds

    def prepare(self, collection: LfmCollection, ui: AnalysisContext) -> LfmInput:
        pri = max(8, round(self.pri_seconds * collection.sample_rate))
        return _input(collection, start=0, count=collection.sample_count("ota"), pri=pri, ui=ui)


def _input(collection: LfmCollection, *, start: int, count: int, pri: int, ui: AnalysisContext) -> LfmInput:
    calibration = ui.once("lfm-calibration-counts", lambda: collection.read("calibration"))
    noise = ui.once("lfm-noise-counts", lambda: collection.read("terminated-noise"))
    return LfmInput(
        sample_rate=collection.sample_rate,
        calibration_dbm=collection.calibration_dbm,
        adc_bits=collection.adc_bits,
        pri_samples=pri,
        start_sample=start,
        calibration_counts=calibration,
        noise_counts=noise,
        ota_counts=collection.read("ota", start, count),
    )


def create_lfm_workspace(
    path: Path | None,
    *,
    identifier: str,
    name: str,
    delivery: BufferedDelivery | WholeFileDelivery,
) -> AnalysisWorkspace:
    directory = path or Path.cwd() / "data" / "lfm-collection"
    return AnalysisWorkspace(
        identifier=identifier,
        name=name,
        description="Manifest-defined calibration, noise, and OTA LFM collection.",
        source=DirectorySource(directory, pattern="*.sigmf-collection", loader=read_collection, describe=describe_collection),
        delivery=delivery,
        analyze=analyze_lfm,
        category="signal analysis",
        tags=("lfm", "10-mhz", "calibration", "four-channel"),
    )


def describe_collection(path: Path) -> DataResource:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return DataResource(
        path.stem,
        payload["collection"]["name"],
        source=path,
        tags=("sigmf-collection", "ci16", "four-channel"),
        summary={"members": "calibration, terminated-noise, ota"},
    )


def read_collection(path: Path) -> LfmCollection:
    payload = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[str, list[CollectionMember]] = {}
    for value in payload["members"]:
        member = CollectionMember(
            value["role"], int(value["channel"]), path.parent / value["metadata"], path.parent / value["data"], float(value["duration_seconds"])
        )
        grouped.setdefault(member.role, []).append(member)
    members = {role: tuple(sorted(values, key=lambda member: member.channel)) for role, values in grouped.items()}
    required = {"calibration", "terminated-noise", "ota"}
    if set(members) != required:
        raise ValueError(f"Collection must define exactly {sorted(required)}")
    sample_rate = float(payload["collection"]["sample_rate"])
    adc_bits = 16
    for role, records in members.items():
        if [record.channel for record in records] != [1, 2, 3, 4]:
            raise ValueError(f"{role} must define channels 1 through 4")
        for member in records:
            metadata = json.loads(member.metadata_path.read_text(encoding="utf-8"))["global"]
            if metadata.get("core:datatype") != "ci16_le" or int(metadata.get("core:num_channels", 0)) != 1:
                raise ValueError(f"{member.metadata_path.name} must be single-channel ci16_le")
            if float(metadata["core:sample_rate"]) != sample_rate:
                raise ValueError(f"{member.metadata_path.name} has a different sample rate")
            expected_bytes = round(member.duration * sample_rate) * 4
            if member.data_path.stat().st_size < expected_bytes:
                raise ValueError(f"{member.data_path.name} is shorter than its declared duration")
    return LfmCollection(sample_rate, float(payload["collection"]["calibration_dbm"]), adc_bits, members)


@dataclass(frozen=True)
class Calibration:
    phase_offsets: np.ndarray
    volts_per_count: np.ndarray
    noise_power_dbm: np.ndarray
    noise_psd_dbm_hz: np.ndarray
    noise_figure_db: np.ndarray
    full_scale_dbm: np.ndarray


@dataclass(frozen=True)
class Products:
    fast_time_us: np.ndarray
    slow_time_s: np.ndarray
    frequencies_hz: np.ndarray
    time_mean_dbm: np.ndarray
    time_max_dbm: np.ndarray
    time_waterfall_dbm: np.ndarray
    psd_mean_dbm_hz: np.ndarray
    psd_max_dbm_hz: np.ndarray
    psd_waterfall_dbm_hz: np.ndarray


def analyze_lfm(data: LfmInput, ui: AnalysisContext) -> None:
    calibration = _calibrate(data)
    ota = _apply_calibration(data.ota_counts, calibration)
    calibrated_tone = _apply_calibration(data.calibration_counts, calibration)
    calibrated_noise = data.noise_counts * calibration.volts_per_count[:, None]
    products = _products(ota, data.sample_rate, data.pri_samples, data.start_sample)

    phase_rows = [
        {"Channel": channel + 1, "Phase correction": f"{-calibration.phase_offsets[channel] * 180 / pi:+.2f} deg"}
        for channel in range(4)
    ]
    amplitude_rows = [
        {
            "Channel": channel + 1,
            "Calibration magnitude": f"{np.sqrt(np.mean(np.abs(data.calibration_counts[channel]) ** 2)):.1f} counts",
            "Scale": f"{calibration.volts_per_count[channel]:.4g} V/count",
            "Full scale": f"{calibration.full_scale_dbm[channel]:.2f} dBm",
        }
        for channel in range(4)
    ]
    noise_rows = [
        {
            "Channel": channel + 1,
            "Noise power": f"{calibration.noise_power_dbm[channel]:.2f} dBm",
            "Noise PSD": f"{calibration.noise_psd_dbm_hz[channel]:.2f} dBm/Hz",
            "Estimated NF": f"{calibration.noise_figure_db[channel]:.2f} dB",
        }
        for channel in range(4)
    ]

    with ui.tab("Phase calibration", columns=(1, 2), update="static"):
        with ui.group("column"):
            ui.text("# Phase calibration\nLearned from the four calibration-tone files.", key="phase-notes")
            ui.table(phase_rows, key="phase-stats")
        with ui.group("column"):
            ui.plot(lambda: _phase_figure(data.calibration_counts, calibration, ui.theme), key="phase-calibration")
    with ui.tab("Amplitude calibration", columns=(1, 2), update="static"):
        with ui.group("column"):
            ui.text(f"# Amplitude calibration\nKnown incident power: {data.calibration_dbm:g} dBm.", key="amplitude-notes")
            ui.table(amplitude_rows, key="amplitude-stats")
        with ui.group("column"):
            ui.plot(lambda: _amplitude_figure(calibrated_tone, data, ui.theme), key="amplitude-calibration")
    with ui.tab("Noise calibration", columns=(1, 2), update="static"):
        with ui.group("column"):
            ui.text("# Terminated-noise calibration\nNoise figure is estimated from the measured noise PSD.", key="noise-notes")
            ui.table(noise_rows, key="noise-stats")
        with ui.group("column"):
            ui.plot(lambda: _noise_figure(calibrated_noise, data, calibration, ui.theme), key="noise-calibration")
    with ui.tab("Waterfall"):
        ui.view_switcher(
            "Domain",
            {
                "Fast-time power": _waterfall_figure(products, "time", ui.theme),
                "Frequency PSD": _waterfall_figure(products, "frequency", ui.theme),
            },
            key="waterfall-domain",
            selector="buttons",
        )
    with ui.tab("Time Domain"):
        ui.plot(_time_figure(products, calibration, ui.theme), key="time")
    with ui.tab("Frequency Domain"):
        ui.plot(_frequency_figure(products, calibration, ui.theme), key="frequency")

    ui.stat("Samples delivered", f"{data.ota_counts.shape[1]:,}")
    ui.stat("Duration delivered", f"{data.ota_counts.shape[1] / data.sample_rate:g} s")
    ui.stat("PRI", f"{data.pri_samples / data.sample_rate:g} s")
    ui.stat("Sample rate", f"{data.sample_rate / 1e6:g} MHz")


def _calibrate(data: LfmInput) -> Calibration:
    reference = data.calibration_counts[0]
    phase_offsets = np.asarray([np.angle(np.mean(channel * np.conj(reference))) for channel in data.calibration_counts])
    desired_voltage = sqrt(2 * R_OHMS * 1e-3 * 10 ** (data.calibration_dbm / 10))
    count_magnitude = np.sqrt(np.mean(np.abs(data.calibration_counts) ** 2, axis=1))
    volts_per_count = desired_voltage / np.maximum(count_magnitude, 1e-12)
    noise_voltage = data.noise_counts * volts_per_count[:, None]
    noise_watts = np.mean(np.abs(noise_voltage) ** 2, axis=1) / (2 * R_OHMS)
    noise_power_dbm = _db10(noise_watts / 1e-3)
    noise_psd = noise_watts / data.sample_rate
    noise_psd_dbm_hz = _db10(noise_psd / 1e-3)
    noise_figure_db = noise_psd_dbm_hz - THERMAL_NOISE_DBM_HZ
    full_scale_voltage = (2 ** (data.adc_bits - 1) - 1) * volts_per_count
    full_scale_dbm = _db10((full_scale_voltage**2 / (2 * R_OHMS)) / 1e-3)
    return Calibration(phase_offsets, volts_per_count, noise_power_dbm, noise_psd_dbm_hz, noise_figure_db, full_scale_dbm)


def _apply_calibration(counts: np.ndarray, calibration: Calibration) -> np.ndarray:
    rotations = np.exp(-1j * calibration.phase_offsets).astype(np.complex64)
    return counts * calibration.volts_per_count[:, None] * rotations[:, None]


def _products(channels: np.ndarray, rate: float, pri: int, start: int, max_rows: int = 384) -> Products:
    row_count = channels.shape[1] // pri
    if row_count < 1:
        raise ValueError("Delivered data must contain at least one PRI")
    rows = channels[:, : row_count * pri].reshape(4, row_count, pri)
    fast_time = np.arange(pri) / rate * 1e6
    power = np.abs(rows) ** 2 / (2 * R_OHMS)
    time_mean = _db10(np.mean(power, axis=1) / 1e-3)
    time_max = _db10(np.max(power, axis=1) / 1e-3)

    nfft = min(pri, 256)
    frequencies = np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / rate))
    frequency_bin_hz = rate / nfft
    psd_sum = np.zeros((4, nfft), dtype=np.float64)
    psd_max = np.zeros((4, nfft), dtype=np.float64)
    time_waterfall = []
    psd_waterfall = []
    slow_time = []
    group_size = max(1, ceil(row_count / max_rows))
    for first in range(0, row_count, group_size):
        block = rows[:, first : min(first + group_size, row_count)]
        block_power = np.abs(block) ** 2 / (2 * R_OHMS)
        time_waterfall.append(_db10(np.mean(block_power, axis=1) / 1e-3))
        spectrum = np.fft.fftshift(np.fft.fft(block[:, :, :nfft], axis=2), axes=2)
        psd = np.abs(spectrum) ** 2 / nfft**2 / (2 * R_OHMS) / frequency_bin_hz
        psd_sum += np.sum(psd, axis=1)
        psd_max = np.maximum(psd_max, np.max(psd, axis=1))
        psd_waterfall.append(_db10(np.mean(psd, axis=1) / 1e-3))
        slow_time.append((start + (first + block.shape[1] / 2) * pri) / rate)
    psd_mean = _db10((psd_sum / row_count) / 1e-3)
    psd_hold = _db10(psd_max / 1e-3)
    return Products(
        fast_time,
        np.asarray(slow_time),
        frequencies,
        time_mean,
        time_max,
        np.stack(time_waterfall, axis=1),
        psd_mean,
        psd_hold,
        np.stack(psd_waterfall, axis=1),
    )


def _db10(value: Any) -> np.ndarray:
    return 10 * np.log10(np.maximum(value, 1e-30))


def _phase_figure(counts: np.ndarray, calibration: Calibration, theme: str) -> go.Figure:
    figure = make_subplots(rows=2, cols=2, specs=[[{"colspan": 2}, None], [{}, {}]], subplot_titles=("Amplitude", "Phase before", "Phase aligned"))
    step = max(1, counts.shape[1] // 512)
    subset = counts[:, ::step][:, :512]
    aligned = subset * np.exp(-1j * calibration.phase_offsets)[:, None]
    for channel in range(4):
        name = f"Channel {channel + 1}"
        figure.add_trace(go.Scatter(y=np.abs(subset[channel]), name=name), row=1, col=1)
        figure.add_trace(go.Scatter(y=np.unwrap(np.angle(subset[channel])), name=name, showlegend=False), row=2, col=1)
        figure.add_trace(go.Scatter(y=np.unwrap(np.angle(aligned[channel])), name=name, showlegend=False), row=2, col=2)
    return style_plotly(figure, title="Phase calibration", theme=theme)


def _amplitude_figure(channels: np.ndarray, data: LfmInput, theme: str) -> go.Figure:
    figure = make_subplots(rows=2, cols=1, subplot_titles=("Signal power", "Signal PSD"))
    step = max(1, channels.shape[1] // 1024)
    subset = channels[:, ::step][:, :1024]
    for channel in range(4):
        power = _db10((np.abs(subset[channel]) ** 2 / (2 * R_OHMS)) / 1e-3)
        frequency, psd = _single_psd(subset[channel], data.sample_rate)
        figure.add_trace(go.Scatter(y=power, name=f"Channel {channel + 1}"), row=1, col=1)
        figure.add_trace(go.Scatter(x=frequency, y=psd, name=f"Channel {channel + 1}", showlegend=False), row=2, col=1)
    figure.add_trace(go.Scatter(x=[0, subset.shape[1] - 1], y=[data.calibration_dbm] * 2, name="Incident power", line={"color": ORANGE, "dash": "dash"}), row=1, col=1)
    return style_plotly(figure, title="Amplitude calibration", theme=theme)


def _noise_figure(channels: np.ndarray, data: LfmInput, calibration: Calibration, theme: str) -> go.Figure:
    figure = make_subplots(rows=2, cols=1, subplot_titles=("Noise power", "Noise PSD"))
    step = max(1, channels.shape[1] // 1024)
    subset = channels[:, ::step][:, :1024]
    for channel in range(4):
        power = _db10((np.abs(subset[channel]) ** 2 / (2 * R_OHMS)) / 1e-3)
        frequency, psd = _single_psd(subset[channel], data.sample_rate)
        figure.add_trace(go.Scatter(y=power, name=f"Channel {channel + 1}"), row=1, col=1)
        figure.add_trace(go.Scatter(x=frequency, y=psd, name=f"Channel {channel + 1}", showlegend=False), row=2, col=1)
    for channel in range(4):
        figure.add_trace(go.Scatter(x=[-data.sample_rate / 2, data.sample_rate / 2], y=[calibration.noise_psd_dbm_hz[channel]] * 2, name=f"Ch {channel + 1} measured floor", line={"dash": "dot"}), row=2, col=1)
    return style_plotly(figure, title="Terminated-noise calibration", theme=theme)


def _single_psd(samples: np.ndarray, rate: float) -> tuple[np.ndarray, np.ndarray]:
    nfft = min(1024, samples.size)
    spectrum = np.fft.fftshift(np.fft.fft(samples[:nfft]))
    psd = np.abs(spectrum) ** 2 / nfft**2 / (2 * R_OHMS) / (rate / nfft)
    return np.fft.fftshift(np.fft.fftfreq(nfft, d=1 / rate)), _db10(psd / 1e-3)


def _waterfall_figure(products: Products, domain: str, theme: str) -> go.Figure:
    figure = make_subplots(rows=2, cols=2, subplot_titles=[f"Channel {channel + 1}" for channel in range(4)])
    for channel in range(4):
        if domain == "time":
            x, z, title = products.fast_time_us, products.time_waterfall_dbm[channel], "Power (dBm)"
        else:
            x, z, title = products.frequencies_hz, products.psd_waterfall_dbm_hz[channel], "PSD (dBm/Hz)"
        figure.add_trace(go.Heatmap(x=x, y=products.slow_time_s, z=z, colorscale="Viridis", showscale=channel == 3, colorbar={"title": title}), row=channel // 2 + 1, col=channel % 2 + 1)
    figure.update_yaxes(title_text="Recording time (s)", col=1)
    figure.update_xaxes(title_text="Fast time (us)" if domain == "time" else "Frequency (Hz)", row=2)
    return style_plotly(figure, title="Fast-time power waterfall" if domain == "time" else "Frequency PSD waterfall", theme=theme)


def _time_figure(products: Products, calibration: Calibration, theme: str) -> go.Figure:
    figure = make_subplots(rows=2, cols=2, subplot_titles=[f"Channel {channel + 1}" for channel in range(4)])
    for channel in range(4):
        row, col = channel // 2 + 1, channel % 2 + 1
        x = products.fast_time_us
        traces = (
            (products.time_mean_dbm[channel], "Mean", TEAL, "solid"),
            (products.time_max_dbm[channel], "Max hold", ORANGE, "solid"),
            (np.full(x.size, calibration.noise_power_dbm[channel]), "Noise power", "#8f9fa6", "dot"),
            (np.full(x.size, calibration.full_scale_dbm[channel]), "Full scale", "#60717d", "dash"),
        )
        for y, name, color, dash in traces:
            figure.add_trace(go.Scatter(x=x, y=y, name=name, line={"color": color, "dash": dash}, showlegend=channel == 0), row=row, col=col)
    figure.update_xaxes(title_text="Fast time (us)", row=2)
    figure.update_yaxes(title_text="Power (dBm)", col=1)
    return style_plotly(figure, title="Fast-time mean and max hold", theme=theme)


def _frequency_figure(products: Products, calibration: Calibration, theme: str) -> go.Figure:
    figure = make_subplots(rows=2, cols=2, subplot_titles=[f"Channel {channel + 1}" for channel in range(4)])
    for channel in range(4):
        row, col = channel // 2 + 1, channel % 2 + 1
        x = products.frequencies_hz
        traces = (
            (products.psd_mean_dbm_hz[channel], "Average", TEAL, "solid"),
            (products.psd_max_dbm_hz[channel], "Max hold", ORANGE, "solid"),
            (np.full(x.size, calibration.noise_psd_dbm_hz[channel]), "Noise PSD", "#8f9fa6", "dot"),
            (np.full(x.size, calibration.full_scale_dbm[channel]), "Full scale", "#60717d", "dash"),
        )
        for y, name, color, dash in traces:
            figure.add_trace(go.Scatter(x=x, y=y, name=name, line={"color": color, "dash": dash}, showlegend=channel == 0), row=row, col=col)
    figure.update_xaxes(title_text="Frequency (Hz)", row=2)
    figure.update_yaxes(title_text="PSD (dBm/Hz)", col=1)
    return style_plotly(figure, title="Average and max-hold PSD", theme=theme)
