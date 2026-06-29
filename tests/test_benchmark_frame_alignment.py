import unittest

import pandas as pd

from src.benchmark.frame_alignment import remap_prediction_frames_to_reference


class TestFrameAlignment(unittest.TestCase):
    def test_remap_prediction_frames_to_reference_space(self) -> None:
        df = pd.DataFrame(
            {
                "abs_frame": [9, 10, 11],
                "rel_frame": [9, 10, 11],
                "global_id": [1, 1, 1],
                "class_id": [1, 1, 1],
            }
        )

        remapped = remap_prediction_frames_to_reference(df, frame_offset=383)

        self.assertEqual(remapped["abs_frame"].tolist(), [392, 393, 394])
        self.assertEqual(remapped["rel_frame"].tolist(), [392, 393, 394])


if __name__ == "__main__":
    unittest.main()
