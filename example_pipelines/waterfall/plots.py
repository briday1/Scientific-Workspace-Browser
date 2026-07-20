"""Plotly construction and view layout for analyzed waterfall products."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sigvue.plugin import Presentation, ViewContext

from ..style import TEAL, style_figure
from .models import WaterfallProducts


COLORMAPS = ("Plasma", "Viridis", "Cividis", "Inferno", "Magma", "Turbo")


def present(products: WaterfallProducts, ui: ViewContext) -> None:
    colormap = ui.colormap(
        "colormap",
        label="Waterfall colormap",
        default="Plasma",
        options=COLORMAPS,
        group="Display",
    )
    finite = products.waterfall_dbfs[np.isfinite(products.waterfall_dbfs)]
    automatic = (
        float(np.floor(np.percentile(finite, 3))) if finite.size else -90.0,
        float(np.ceil(np.percentile(finite, 99.5))) if finite.size else -20.0,
    )
    zmin, zmax = ui.limits(
        "dbfs_limits",
        label="dBFS limits",
        default=automatic,
        minimum=-140.0,
        maximum=0.0,
        step=1.0,
        group="Display",
    )
    spectrum_style = ui.trace_style(
        "spectrum_style",
        label="Average spectrum",
        color=TEAL,
        width=1.4,
        group="Display",
    )
    show_colorbar = ui.toggle(
        "show_colorbar",
        label="Show colorbar",
        default=True,
        group="Display",
    )
    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=(0.12, 0.88),
        vertical_spacing=0.04,
    )
    figure.add_trace(go.Scatter(
        x=products.frequency_mhz,
        y=products.spectrum_dbfs,
        mode=spectrum_style.mode,
        line=spectrum_style.line,
        marker=spectrum_style.plotly_marker,
        name="Average spectrum",
    ), row=1, col=1)
    figure.add_trace(go.Heatmap(
        x=products.frequency_mhz,
        y=products.time_edges_ms,
        z=products.waterfall_dbfs,
        zmin=zmin,
        zmax=zmax,
        colorscale=colormap,
        showscale=show_colorbar,
        colorbar={"title": "dBFS"},
    ), row=2, col=1)
    figure.update_yaxes(title_text="Power (dBFS)", range=[zmin, zmax], autorange=False, row=1, col=1)
    figure.update_yaxes(
        title_text="Recording time (ms)",
        range=[float(products.time_edges_ms[0]), float(products.time_edges_ms[-1])],
        autorange=False,
        row=2,
        col=1,
    )
    figure.update_xaxes(title_text="RF frequency (MHz)", row=2, col=1)
    figure.update_layout(uirevision=f"lte-waterfall:{products.recording.metadata_path}")
    title = str(products.recording.metadata["global"].get("core:description", "Synthetic LTE"))
    ui.stat("Sample rate", f"{products.recording.sample_rate / 1e6:g} MS/s")
    ui.stat("Center frequency", f"{products.recording.center_frequency / 1e6:g} MHz")
    with ui.tab("Spectrum + waterfall"):
        ui.plot(style_figure(figure, ui.theme, title), key="lte-waterfall", axis_navigation="bounded")


class WaterfallPresentation(Presentation[WaterfallProducts]):
    """Framework presentation object for the waterfall views."""

    def present(self, products: WaterfallProducts, ui: ViewContext) -> None:
        present(products, ui)
