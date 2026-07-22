"""Resolution-bounded, aggregation-aware Plotly heatmap rendering."""

from __future__ import annotations

import base64
import warnings
from dataclasses import dataclass
from io import BytesIO
from math import ceil
from typing import Any, Literal

import numpy as np
import plotly.colors as plotly_colors
import plotly.graph_objects as go
from PIL import Image


HeatmapAggregation = Literal["max", "mean", "median"]
HEATMAP_AGGREGATIONS: tuple[HeatmapAggregation, ...] = ("max", "mean", "median")
_LUT_SIZE = 256


def aggregate_heatmap(
    values: Any,
    *,
    width: int,
    height: int,
    method: HeatmapAggregation = "mean",
) -> np.ndarray:
    """Reduce a 2-D matrix into at most ``width`` by ``height`` exact blocks."""
    matrix = np.asarray(values)
    if matrix.ndim != 2:
        raise ValueError(f"Heatmap values must be 2-D, received shape {matrix.shape}")
    if width <= 0 or height <= 0:
        raise ValueError("Heatmap render width and height must be positive")
    if method not in HEATMAP_AGGREGATIONS:
        raise ValueError(f"Unsupported heatmap aggregation: {method}")
    rows, columns = matrix.shape
    if rows == 0 or columns == 0:
        raise ValueError("Heatmap values cannot be empty")
    row_block = max(1, ceil(rows / height))
    column_block = max(1, ceil(columns / width))
    output_rows = ceil(rows / row_block)
    output_columns = ceil(columns / column_block)
    padded = np.full(
        (output_rows * row_block, output_columns * column_block),
        np.nan,
        dtype=np.result_type(matrix.dtype, np.float32),
    )
    padded[:rows, :columns] = matrix
    blocks = padded.reshape(output_rows, row_block, output_columns, column_block)
    reducer = {"max": np.nanmax, "mean": np.nanmean, "median": np.nanmedian}[method]
    with warnings.catch_warnings(), np.errstate(invalid="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.asarray(reducer(blocks, axis=(1, 3)))


def _colorscale_lut(colorscale: Any) -> np.ndarray:
    if not isinstance(colorscale, str):
        colorscale = [list(stop) for stop in colorscale]
    colors = plotly_colors.sample_colorscale(
        colorscale,
        np.linspace(0.0, 1.0, _LUT_SIZE),
        colortype="rgb",
    )
    return np.asarray([plotly_colors.unlabel_rgb(color) for color in colors], dtype=np.uint8)


def _png_uri(values: np.ndarray, *, zmin: float, zmax: float, colorscale: Any) -> str:
    zmin = float(zmin)
    zmax = float(zmax)
    if not np.isfinite(zmin) or not np.isfinite(zmax) or zmax <= zmin:
        raise ValueError(f"Heatmap requires finite zmax > zmin, received {zmin}, {zmax}")
    finite = np.isfinite(values)
    scaled = np.zeros(values.shape, dtype=np.float32)
    np.subtract(values, zmin, out=scaled, where=finite)
    scaled *= (_LUT_SIZE - 1) / (zmax - zmin)
    np.clip(scaled, 0.0, _LUT_SIZE - 1, out=scaled)
    indices = np.rint(scaled).astype(np.uint8)
    rgba = np.empty((*values.shape, 4), dtype=np.uint8)
    rgba[..., :3] = _colorscale_lut(colorscale)[indices]
    rgba[..., 3] = np.where(finite, 255, 0).astype(np.uint8)
    rgba = np.ascontiguousarray(np.flipud(rgba))
    buffer = BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(
        buffer,
        format="PNG",
        compress_level=1,
        optimize=False,
    )
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode('ascii')}"


def _coordinate_bounds(values: Any, cell_count: int) -> tuple[float, float]:
    if values is None:
        return -0.5, float(cell_count) - 0.5
    coordinates = np.asarray(values, dtype=float)
    if coordinates.ndim != 1 or coordinates.size == 0:
        raise ValueError("Rasterized heatmap coordinates must be one-dimensional and non-empty")
    if coordinates.size == cell_count + 1:
        return float(coordinates[0]), float(coordinates[-1])
    if coordinates.size == 1:
        return float(coordinates[0] - 0.5), float(coordinates[0] + 0.5)
    spacing = np.diff(coordinates)
    return float(coordinates[0] - spacing[0] / 2), float(coordinates[-1] + spacing[-1] / 2)


@dataclass(frozen=True)
class RasterizedHeatmap:
    """Wrap a Plotly Heatmap and render its matrix as one bounded PNG image."""

    trace: go.Heatmap
    render_width: int = 1024
    render_height: int = 512
    aggregation: HeatmapAggregation = "mean"

    def __post_init__(self) -> None:
        if not isinstance(self.trace, go.Heatmap):
            raise TypeError("RasterizedHeatmap requires a plotly.graph_objects.Heatmap")
        if self.render_width <= 0 or self.render_height <= 0:
            raise ValueError("Heatmap render width and height must be positive")
        if self.aggregation not in HEATMAP_AGGREGATIONS:
            raise ValueError(f"Unsupported heatmap aggregation: {self.aggregation}")

    @classmethod
    def create(
        cls,
        *,
        render_width: int = 1024,
        render_height: int = 512,
        aggregation: HeatmapAggregation = "mean",
        **heatmap: Any,
    ) -> "RasterizedHeatmap":
        """Construct a wrapped Heatmap from normal Plotly keyword arguments."""
        return cls(go.Heatmap(**heatmap), render_width, render_height, aggregation)

    @property
    def render_shape(self) -> tuple[int, int]:
        rows, columns = np.asarray(self.trace.z).shape
        row_block = max(1, ceil(rows / self.render_height))
        column_block = max(1, ceil(columns / self.render_width))
        return ceil(rows / row_block), ceil(columns / column_block)

    def add_to(self, figure: go.Figure, *, row: int | None = None, col: int | None = None) -> int:
        """Add the raster and its lightweight colorbar carrier to ``figure``."""
        trace = go.Heatmap(self.trace)
        original = np.asarray(trace.z)
        rendered = aggregate_heatmap(
            original,
            width=self.render_width,
            height=self.render_height,
            method=self.aggregation,
        )
        finite = original[np.isfinite(original)]
        if finite.size == 0:
            raise ValueError("Heatmap values must contain at least one finite value")
        zmin = float(trace.zmin) if trace.zmin is not None else float(np.min(finite))
        zmax = float(trace.zmax) if trace.zmax is not None else float(np.max(finite))
        if zmax <= zmin:
            zmax = zmin + max(1.0, abs(zmin) * 1e-9)
        x_bounds = _coordinate_bounds(trace.x, original.shape[1])
        y_bounds = _coordinate_bounds(trace.y, original.shape[0])
        if row is None and col is None:
            figure.add_trace(trace)
        elif row is not None and col is not None:
            figure.add_trace(trace, row=row, col=col)
        else:
            raise ValueError("Heatmap subplot row and col must be supplied together")
        trace_index = len(figure.data) - 1
        attached = figure.data[trace_index]
        xref = attached.xaxis or "x"
        yref = attached.yaxis or "y"
        xmin, xmax = x_bounds
        ymin, ymax = y_bounds
        figure.add_layout_image(
            source=_png_uri(rendered, zmin=zmin, zmax=zmax, colorscale=trace.colorscale),
            name=trace.name,
            xref=xref,
            yref=yref,
            x=xmin,
            y=ymax,
            sizex=xmax - xmin,
            sizey=ymax - ymin,
            xanchor="left",
            yanchor="top",
            sizing="stretch",
            opacity=1.0,
            layer="below",
            visible=trace.visible is not False and trace.visible != "legendonly",
        )
        attached.x = [xmin, xmax]
        attached.y = [ymin, ymax]
        attached.z = [[zmin, zmax], [zmin, zmax]]
        attached.opacity = 0.0
        attached.hoverinfo = "skip"
        attached.hovertemplate = None
        return trace_index


__all__ = [
    "HEATMAP_AGGREGATIONS",
    "HeatmapAggregation",
    "RasterizedHeatmap",
    "aggregate_heatmap",
]
