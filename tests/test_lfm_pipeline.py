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
    WholeFileDelivery,
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

    def test_whole_file_delivery_has_no_controls_and_returns_all_samples(self):
        with TemporaryDirectory() as directory:
            collection = self._collection(Path(directory), sample_count=1_000)
            ui = AnalysisContext({})
            delivered = WholeFileDelivery(pri_seconds=0.01).prepare(collection, ui)
            self.assertEqual((4, 1_000), delivered.ota_counts.shape)
            self.assertEqual([], ui.controls)
            self.assertFalse(ui.playback_config.enabled)

    def test_buffered_delivery_owns_playback_and_returns_only_window(self):
        with TemporaryDirectory() as directory:
            collection = self._collection(Path(directory), sample_count=1_000)
            ui = AnalysisContext({"buffer_seconds": "0.1", "pri_seconds": "0.01", "__playback_time_seconds": "0.4"})
            delivered = BufferedDelivery().prepare(collection, ui)
            self.assertEqual((4, 100), delivered.ota_counts.shape)
            self.assertEqual(400, delivered.start_sample)
            self.assertEqual(["buffer_seconds", "pri_seconds", "seek_seconds", "refresh_seconds"], [control.name for control in ui.controls])
            self.assertTrue(ui.playback_config.enabled)

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
