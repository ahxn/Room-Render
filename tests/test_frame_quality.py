import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from room_reconstruction.errors import FrameQualityError
from room_reconstruction.frame_quality import calculate_blur_score, filter_frames


class FrameQualityTests(unittest.TestCase):
    def test_load_failure_is_actionable(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            self.assertRaisesRegex(FrameQualityError, "Could not load extracted frame"),
        ):
            calculate_blur_score(Path(temporary_directory) / "missing.jpg")

    def test_sharp_image_scores_higher_than_blurred_image(self) -> None:
        import cv2

        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            sharp = np.indices((100, 100)).sum(axis=0) % 2 * 255
            sharp = sharp.astype(np.uint8)
            blurred = cv2.GaussianBlur(sharp, (21, 21), 0)
            sharp_path = root / "sharp.jpg"
            blurred_path = root / "blurred.jpg"
            cv2.imwrite(str(sharp_path), sharp)
            cv2.imwrite(str(blurred_path), blurred)

            self.assertGreater(calculate_blur_score(sharp_path), calculate_blur_score(blurred_path))

    def test_threshold_selects_frames_and_writes_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            frames = root / "frames"
            frames.mkdir()
            first = frames / "frame_00001.jpg"
            second = frames / "frame_00002.jpg"
            first.write_bytes(b"first")
            second.write_bytes(b"second")
            selected = root / "selected"
            manifest = root / "frame_quality.json"

            with patch(
                "room_reconstruction.frame_quality.calculate_blur_score",
                side_effect=[50.0, 150.0],
            ):
                result = filter_frames(
                    frames,
                    selected,
                    manifest,
                    minimum_blur_score=100.0,
                )

            self.assertEqual(result.accepted_count, 1)
            self.assertEqual(result.rejected_count, 1)
            self.assertFalse((selected / first.name).exists())
            self.assertTrue((selected / second.name).exists())
            payload = json.loads(manifest.read_text(encoding="utf-8"))
            self.assertEqual(payload["accepted_count"], 1)
            self.assertEqual([item["accepted"] for item in payload["frames"]], [False, True])
