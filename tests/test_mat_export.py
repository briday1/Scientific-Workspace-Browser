import unittest
from dataclasses import dataclass
from io import BytesIO

import numpy as np
import plotly.graph_objects as go
from scipy.io import loadmat

from workspace_browser.web.mat_export import write_mat_export


@dataclass
class DeliveredBuffer:
    start_sample: int
    samples: np.ndarray


class MatExportTests(unittest.TestCase):
    def test_export_contains_delivered_buffer_parameters_and_plot_traces(self):
        samples = np.asarray([1 + 2j, 3 + 4j], dtype=np.complex64)
        figure = go.Figure(go.Scatter(x=[0.0, 0.5], y=[-20.0, -18.0], name="Power"))
        stream = BytesIO()

        write_mat_export(
            stream,
            workspace_id="signals",
            item_id="capture-1",
            playback={"mode": "seek", "duration_seconds": 4.0},
            refresh={"enabled": False},
            parameters={"__playback_time_seconds": 1.25, "buffer_seconds": 0.5},
            controls=[{"name": "buffer_seconds", "control_type": "float"}],
            delivered_data=DeliveredBuffer(start_sample=100, samples=samples),
            layout={"kind": "tabs", "children": ["Power", "Diagnostics"]},
            metadata={"sample_rate": 2.0},
            statistics={"Samples": 2},
            views=[("Power plot", figure), ("Diagnostics", [{"Peak": -18.0}])],
        )

        self.assertEqual(b"MATL", stream.getvalue()[:4])
        stream.seek(0)
        exported = loadmat(stream, simplify_cells=True)["workspace_export"]
        self.assertEqual("signals", exported["workspace_id"])
        self.assertEqual("seek", exported["playback"]["mode"])
        self.assertEqual(100, exported["data"]["start_sample"])
        np.testing.assert_allclose(samples, exported["data"]["samples"])
        trace = exported["views"]["view_001_Power_plot"]["traces"]["trace_001"]
        np.testing.assert_allclose([0.0, 0.5], trace["x"])
        np.testing.assert_allclose([-20.0, -18.0], trace["y"])
        self.assertEqual("value", exported["views"]["view_002_Diagnostics"]["kind"])
        self.assertEqual("tabs", exported["layout"]["kind"])


if __name__ == "__main__":
    unittest.main()
