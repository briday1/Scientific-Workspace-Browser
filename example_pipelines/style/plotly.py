"""Small Plotly theme helper kept independent of pipeline calculations."""

from typing import Any


ACCENT = "#7c5cff"


def style_figure(figure: Any, theme: str, title: str) -> Any:
    dark = theme == "dark"
    grid = "#34434f" if dark else "#d9e0e5"
    figure.update_layout(
        template="plotly_dark" if dark else "simple_white",
        paper_bgcolor="#111a22" if dark else "#ffffff",
        plot_bgcolor="#111a22" if dark else "#ffffff",
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 15}},
        margin={"l": 70, "r": 30, "t": 58, "b": 52},
    )
    figure.update_xaxes(showgrid=True, gridcolor=grid, showline=True, mirror=True, linecolor=grid, zeroline=False)
    figure.update_yaxes(showgrid=True, gridcolor=grid, showline=True, mirror=True, linecolor=grid, zeroline=False)
    return figure
