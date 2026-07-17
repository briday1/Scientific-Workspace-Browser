import unittest
from io import BytesIO
from zipfile import ZipFile

import matplotlib.pyplot as plt

from workspace_browser.web.png_export import write_png_bundle


class PngExportTests(unittest.TestCase):
    def test_bundle_contains_every_plot_and_skips_non_plot_views(self):
        first, first_axis = plt.subplots()
        first_axis.plot([0, 1], [1, 2])
        second, second_axis = plt.subplots()
        second_axis.imshow([[0, 1], [2, 3]])
        stream = BytesIO()
        try:
            count = write_png_bundle(
                stream,
                [("Time plot", first), ("Diagnostics", [{"Peak": 2}]), ("Waterfall", second)],
                filename_prefix="capture-t0.125s-buffer0.02s-live",
            )
        finally:
            plt.close(first)
            plt.close(second)

        self.assertEqual(2, count)
        stream.seek(0)
        with ZipFile(stream) as archive:
            names = archive.namelist()
            self.assertEqual(
                [
                    "001-capture-t0.125s-buffer0.02s-live-Time-plot.png",
                    "003-capture-t0.125s-buffer0.02s-live-Waterfall.png",
                    "README.txt",
                ],
                names,
            )
            self.assertEqual(
                b"\x89PNG\r\n\x1a\n",
                archive.read("001-capture-t0.125s-buffer0.02s-live-Time-plot.png")[:8],
            )


if __name__ == "__main__":
    unittest.main()
