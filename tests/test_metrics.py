import json
import unittest

from room_reconstruction.errors import RegistrationError
from room_reconstruction.metrics import count_registered_frames, registration_rate


class MetricsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = __import__("tempfile").TemporaryDirectory()
        self.root = __import__("pathlib").Path(self.temporary_directory.name)

    def tearDown(self) -> None:
        self.temporary_directory.cleanup()

    def test_count_registered_frames(self) -> None:
        transforms = self.root / "transforms.json"
        transforms.write_text(json.dumps({"frames": [{}, {}, {}]}), encoding="utf-8")
        self.assertEqual(count_registered_frames(transforms), 3)

    def test_count_registered_frames_rejects_invalid_shape(self) -> None:
        transforms = self.root / "transforms.json"
        transforms.write_text(json.dumps({"frames": "not-a-list"}), encoding="utf-8")
        with self.assertRaisesRegex(RegistrationError, "frame list"):
            count_registered_frames(transforms)

    def test_registration_rate(self) -> None:
        self.assertAlmostEqual(registration_rate(189, 202), 0.93564356)

    def test_registration_rate_rejects_inconsistent_counts(self) -> None:
        for registered, selected in [(-1, 2), (3, 2), (0, 0)]:
            with self.subTest(registered=registered, selected=selected), self.assertRaises(
                ValueError
            ):
                registration_rate(registered, selected)
