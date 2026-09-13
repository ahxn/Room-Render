import tempfile
import unittest
from pathlib import Path

from room_reconstruction.config import load_pipeline_config
from room_reconstruction.errors import ConfigurationError


class ConfigurationTests(unittest.TestCase):
    def test_loads_nested_yaml_configuration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.yml"
            path.write_text(
                """
video:
  target_frame_count: 180
frame_quality:
  enabled: true
  minimum_blur_score: 3.0
reconstruction:
  max_num_iterations: 15000
  cache_images: cpu
  camera_res_scale_factor: 0.5
""".strip(),
                encoding="utf-8",
            )

            config = load_pipeline_config(path)

            self.assertEqual(config.video.target_frame_count, 180)
            self.assertTrue(config.frame_quality.enabled)
            self.assertEqual(config.frame_quality.minimum_blur_score, 3.0)
            self.assertEqual(config.reconstruction.max_num_iterations, 15000)
            self.assertEqual(config.reconstruction.cache_images, "cpu")

    def test_unknown_option_fails_with_section_name(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.yml"
            path.write_text("video:\n  mystery_option: 1\n", encoding="utf-8")

            with self.assertRaisesRegex(ConfigurationError, "mystery_option"):
                load_pipeline_config(path)

    def test_invalid_low_memory_value_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.yml"
            path.write_text(
                "reconstruction:\n  camera_res_scale_factor: 0\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ConfigurationError, "camera_res_scale_factor"):
                load_pipeline_config(path)

    def test_wrong_scalar_type_fails_with_actionable_error(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.yml"
            path.write_text('video:\n  target_frame_count: "many"\n', encoding="utf-8")

            with self.assertRaisesRegex(ConfigurationError, "target_frame_count"):
                load_pipeline_config(path)
