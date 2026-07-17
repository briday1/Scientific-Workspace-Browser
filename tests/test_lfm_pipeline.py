import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from workspace_browser.core.plugin import AnalysisContext
from workspace_browser.examples.lfm_collection import create_workspace as create_buffered_workspace
from workspace_browser.examples.lfm_full_recording import create_workspace as create_full_workspace
from workspace_browser.examples.lfm_pipeline import (
    BufferedDelivery,
    CollectionMember,
    LfmCollection,
    LfmInput,
    WholeFileDelivery,
    _products,
    analyze_lfm,
)


class LfmDeliveryTests(unittest.TestCase):
    def test_workspace_entry_points_share_pipeline_and_change_only_delivery(self):
        buffered = create_buffered_workspace(Path("missing"))
        full = create_full_workspace(Path("missing"))
        self.assertIs(analyze_lfm, buffered.analyze)
        self.assertIs(analyze_lfm, full.analyze)
        self.assertIsInstance(buffered.delivery, BufferedDelivery)
        self.assertIsInstance(full.delivery, WholeFileDelivery)

    def test_whole_file_delivery_has_only_processing_prf_and_returns_all_samples(self):
        with TemporaryDirectory() as directory:
            collection = self._collection(Path(directory), sample_count=1_000)
            ui = AnalysisContext({})
            delivered = WholeFileDelivery(default_processing_prf_hz=100).prepare(collection, ui)
            self.assertEqual((4, 1_000), delivered.ota_counts.shape)
            self.assertEqual(10, delivered.pri_samples)
            self.assertEqual(["processing_prf_hz"], [control.name for control in ui.controls])
            self.assertEqual("static", ui.playback_config.mode)

    def test_buffered_delivery_owns_playback_and_returns_only_window(self):
        with TemporaryDirectory() as directory:
            collection = self._collection(Path(directory), sample_count=1_000)
            ui = AnalysisContext({"buffer_seconds": "0.1", "processing_prf_hz": "100", "__playback_time_seconds": "0.4"})
            delivered = BufferedDelivery().prepare(collection, ui)
            self.assertEqual((4, 100), delivered.ota_counts.shape)
            self.assertEqual(400, delivered.start_sample)
            self.assertEqual(10, delivered.pri_samples)
            self.assertEqual(["buffer_seconds", "processing_prf_hz", "seek_seconds", "refresh_seconds"], [control.name for control in ui.controls])
            self.assertEqual("live", ui.playback_config.mode)

    def test_buffered_delivery_can_expose_seek_without_live_controls(self):
        with TemporaryDirectory() as directory:
            collection = self._collection(Path(directory), sample_count=1_000)
            ui = AnalysisContext({"buffer_seconds": "0.1", "processing_prf_hz": "100"})
            BufferedDelivery(playback_mode="seek").prepare(collection, ui)
            self.assertEqual("seek", ui.playback_config.mode)

    def test_live_tail_rechecks_common_file_growth_and_preserves_historical_seek(self):
        with TemporaryDirectory() as directory:
            collection = self._collection(Path(directory), sample_count=1_000)
            live_values = {
                "buffer_seconds": "0.1",
                "processing_prf_hz": "100",
                "__playback_follow_live": "true",
            }
            initial_ui = AnalysisContext(live_values)
            initial = BufferedDelivery().prepare(collection, initial_ui)
            self.assertEqual(900, initial.start_sample)
            self.assertEqual(0.9, initial_ui.playback_config.duration_seconds)

            extra = np.zeros((200, 2), dtype="<i2")
            for member in collection.members["ota"]:
                with member.data_path.open("ab") as stream:
                    extra.tofile(stream)
            grown = BufferedDelivery().prepare(collection, AnalysisContext(live_values))
            self.assertEqual(1_100, grown.start_sample)

            historical = BufferedDelivery().prepare(
                collection,
                AnalysisContext({**live_values, "__playback_follow_live": "false", "__playback_time_seconds": "0.4"}),
            )
            self.assertEqual(400, historical.start_sample)

    def test_processing_prf_changes_whole_file_reshape_without_changing_data(self):
        with TemporaryDirectory() as directory:
            collection = self._collection(Path(directory), sample_count=1_000)
            delivered = WholeFileDelivery().prepare(collection, AnalysisContext({"processing_prf_hz": "50"}))
            self.assertEqual((4, 1_000), delivered.ota_counts.shape)
            self.assertEqual(20, delivered.pri_samples)

    def test_fractional_pri_start_offsets_fast_time_coordinates(self):
        channels = np.ones((4, 400), dtype=np.complex64)
        aligned = _products(channels, rate=1_000.0, pri=100, start=200)
        shifted = _products(channels, rate=1_000.0, pri=100, start=225)
        self.assertAlmostEqual(0.0, aligned.fast_time_us[0])
        self.assertAlmostEqual(25_000.0, shifted.fast_time_us[0])
        np.testing.assert_allclose(25_000.0, shifted.fast_time_us - aligned.fast_time_us)
        np.testing.assert_allclose(aligned.slow_time_s, shifted.slow_time_s)
        self.assertAlmostEqual(0.05, shifted.slow_time_s[0])

    def test_noise_tab_exercises_inline_number_and_dropdown_parameters(self):
        samples = np.ones((4, 100), dtype=np.complex64) * (100 + 25j)
        data = LfmInput(
            sample_rate=1_000.0,
            calibration_dbm=-20.0,
            adc_bits=16,
            pri_samples=10,
            start_sample=0,
            calibration_counts=samples,
            noise_counts=samples * 0.01,
            ota_counts=samples,
        )
        baseline = AnalysisContext({"reference_noise_psd_dbm_hz": "-174", "noise_reference_lines": "Measured only", "adc_bits": "8"})
        changed = AnalysisContext({"reference_noise_psd_dbm_hz": "-168.5", "noise_reference_lines": "Expected only", "adc_bits": "16"})
        analyze_lfm(data, baseline)
        analyze_lfm(data, changed)

        inline = [control for control in changed.controls if control.placement == "inline"]
        self.assertEqual(["adc_bits", "reference_noise_psd_dbm_hz", "noise_reference_lines"], [control.name for control in inline])
        self.assertEqual(["Waterfall", "Time Domain", "Frequency Domain", "Calibration"], [tab.label for tab in changed.tabs])
        switcher = changed.tabs[-1].nodes[0]
        self.assertEqual("view_switcher", switcher.kind)
        self.assertEqual(["Phase", "Amplitude", "Noise"], [node.props["label"] for node in switcher.children])
        self.assertTrue(all(node.kind == "grid" for node in switcher.children))
        self.assertEqual(["view_slot", "view_slot"], [child.kind for child in switcher.children[0].children])
        self.assertEqual(["column", "view_slot"], [child.kind for child in switcher.children[1].children])
        self.assertEqual(["column", "view_slot"], [child.kind for child in switcher.children[2].children])
        self.assertFalse(any(trace.type == "table" for key, figure in changed.figures.items() if key.endswith("-plot") for trace in figure.data))

        baseline_nf = float(baseline.figures["noise-diagnostics"][0]["Estimated NF"].split()[0])
        changed_nf = float(changed.figures["noise-diagnostics"][0]["Estimated NF"].split()[0])
        self.assertAlmostEqual(5.5, baseline_nf - changed_nf)
        baseline_full_scale = float(baseline.figures["amplitude-diagnostics"][0]["Full scale"].split()[0])
        changed_full_scale = float(changed.figures["amplitude-diagnostics"][0]["Full scale"].split()[0])
        self.assertAlmostEqual(48.23, changed_full_scale - baseline_full_scale, delta=0.02)
        names = [trace.name or "" for trace in changed.figures["noise-plot"].data]
        self.assertIn("Expected noise PSD", names)
        self.assertFalse(any("measured floor" in name for name in names))

    @staticmethod
    def _collection(root: Path, *, sample_count: int) -> LfmCollection:
        members = {}
        for role in ("calibration", "terminated-noise", "ota"):
            records = []
            for channel in range(1, 5):
                path = root / f"{role}-{channel}.sigmf-data"
                iq = np.empty((sample_count, 2), dtype="<i2")
                iq[:, 0] = channel
                iq[:, 1] = -channel
                iq.tofile(path)
                records.append(CollectionMember(role, channel, root / "unused.sigmf-meta", path, 1.0))
            members[role] = tuple(records)
        return LfmCollection(1_000.0, -20.0, 16, members)


if __name__ == "__main__":
    unittest.main()
