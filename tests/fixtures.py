"""Tiny framework-only workspaces used by the test suite."""

from __future__ import annotations

from matplotlib.figure import Figure
import plotly.graph_objects as go

from workspace_browser.plugin import AnalysisWorkspace, DataResource
from workspace_browser.web.application import WorkspaceBrowserApp


class MemorySource:
    def discover(self):
        return [DataResource("recording", "Recording", source=(1.0, 2.0, 3.0, 4.0))]

    def open(self, resource):
        return resource.source


def analyze_plotly(data, ui):
    gain = ui.number("gain", default=1.0, step=0.1)
    ui.playback(mode="seek", duration=2.0, step=0.25)
    with ui.tab("Summary", update="static"):
        ui.text("# Summary\nSynthetic test recording", key="summary")
    with ui.tab("Signal"):
        ui.plot(go.Figure(go.Scatter(y=[gain * value for value in data])), key="signal")


def analyze_matplotlib(data, ui):
    ui.playback(mode="seek", duration=2.0, step=0.25)
    figure = Figure()
    axes = figure.subplots()
    axes.plot(data)
    with ui.tab("Signal"):
        ui.plot(figure, key="signal")


def create_workspace(config=None):
    values = config or {}
    return AnalysisWorkspace(
        identifier=str(values.get("id", "test-workspace")),
        name=str(values.get("name", "Test Workspace")),
        description="Framework test fixture",
        source=MemorySource(),
        analyze=analyze_plotly,
    )


def create_test_app() -> WorkspaceBrowserApp:
    app = WorkspaceBrowserApp(title="Signal Analysis Browser")
    app.register_workspace(create_workspace())
    app.register_workspace(
        AnalysisWorkspace(
            identifier="matplotlib-workspace",
            name="Matplotlib Workspace",
            description="Matplotlib export fixture",
            source=MemorySource(),
            analyze=analyze_matplotlib,
        )
    )
    return app
