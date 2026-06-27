import tempfile
import unittest
from pathlib import Path

from src.benchmark.processed_dataset_loader import discover_processed_dataset, discover_processed_datasets


class ProcessedDatasetLoaderTest(unittest.TestCase):
    def test_discover_latest_timestamp_and_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            video_dir = root / "AIS29T7"
            older = video_dir / "2026-06-01_00-00-00" / "final"
            newer = video_dir / "2026-06-15_00-00-00" / "final"
            older.mkdir(parents=True)
            newer.mkdir(parents=True)

            required_files = [
                "counts.json",
                "run_summary.json",
                "tracks_clean.parquet",
                "tracks.parquet",
                "seg_masks.parquet",
                "tracking.mp4",
            ]
            for path in required_files:
                (older / path).write_text("{}", encoding="utf-8")
                (newer / path).write_text("{}", encoding="utf-8")

            discovery = discover_processed_dataset(root, video_id="AIS29T7")
            self.assertEqual(discovery["video_id"], "AIS29T7")
            self.assertEqual(discovery["timestamp"], "2026-06-15_00-00-00")
            self.assertEqual(discovery["reference_dir"], str(newer))
            self.assertEqual(discovery["missing_files"], [])

            all_datasets = discover_processed_datasets(root)
            self.assertEqual(len(all_datasets), 1)
            self.assertEqual(all_datasets[0]["video_id"], "AIS29T7")


if __name__ == "__main__":
    unittest.main()
