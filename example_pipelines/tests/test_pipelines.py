from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from example_pipelines.comms.workspace import create_workspace as create_comms_workspace
from example_pipelines.scripts.generate_comms import generate as generate_comms
from example_pipelines.scripts.generate_lte import generate as generate_lte
from example_pipelines.waterfall.workspace import create_workspace as create_waterfall_workspace


class ExamplePipelineTests(unittest.TestCase):
    def test_synthetic_lte_generator_and_waterfall_workspace(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "lte"
            generated = generate_lte(root)
            self.assertEqual(2, len(generated))
            self.assertTrue(all(metadata.is_file() and data.is_file() for metadata, data in generated))

            workspace = create_waterfall_workspace({"data_root": root})
            items = workspace.discover_items()
            self.assertEqual(2, len(items))
            self.assertEqual({"downlink", "uplink"}, {item.navigation_path[0] for item in items})

            opened = workspace.open_item(items[0].identifier)
            self.assertEqual("windowed", opened.page.playback.mode)
            self.assertEqual("Mean received power (dBFS)", opened.page.playback.overview_label)
            self.assertEqual(1, len(opened.page.views))
            controls = {control.name: control for control in opened.page.controls}
            self.assertEqual("select", controls["fft_size"].control_type)
            self.assertEqual("select", controls["overlap_percent"].control_type)
            self.assertEqual("colormap", controls["colormap"].control_type)
            self.assertEqual("limits", controls["dbfs_limits"].control_type)
            self.assertEqual("spectrum_style", controls["spectrum_style_color"].picker)
            self.assertEqual("toggle", controls["show_colorbar"].control_type)
            figure = opened.page.views[0].callback({})
            self.assertEqual(["scatter", "heatmap"], [trace.type for trace in figure.data])
            self.assertEqual(figure.data[1].z.shape[0] + 1, len(figure.data[1].y))
            self.assertEqual(tuple(figure.layout.yaxis2.range), (figure.data[1].y[0], figure.data[1].y[-1]))

    def test_synthetic_comms_generator_and_static_workspace(self):
        with TemporaryDirectory() as directory:
            root = Path(directory) / "comms"
            generated = generate_comms(root)
            self.assertEqual(3, len(generated))
            self.assertTrue(all(metadata.is_file() and data.is_file() for metadata, data in generated))

            workspace = create_comms_workspace({"data_root": root})
            items = workspace.discover_items()
            self.assertEqual(3, len(items))
            self.assertEqual(
                {"Synthetic QPSK", "Synthetic 16-QAM", "Synthetic 64-QAM"},
                {item.title for item in items},
            )

            for item in items:
                opened = workspace.open_item(item.identifier)
                self.assertEqual("static", opened.page.playback.mode)
                self.assertEqual(["constellation", "eye"], [view.name for view in opened.page.views])
                figures = [view.callback({}) for view in opened.page.views]
                self.assertEqual("scattergl", figures[0].data[0].type)
                self.assertEqual(["scattergl", "scattergl"], [trace.type for trace in figures[1].data])
