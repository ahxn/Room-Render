import unittest
from pathlib import Path
from unittest.mock import patch

from room_reconstruction.cli import main
from room_reconstruction.config import PipelineConfig


class CliQualityTests(unittest.TestCase):
    def test_low_quality_loads_low_quality_preset(self) -> None:
        with (
            patch(
                "room_reconstruction.cli.load_pipeline_config",
                return_value=PipelineConfig(),
            ) as load_config,
            patch("room_reconstruction.cli.run_pipeline"),
        ):
            result = main(["room.mp4", "--output", "results/room", "--quality", "low"])

        self.assertEqual(result, 0)
        self.assertEqual(load_config.call_args.args[0].name, "low-quality.yml")

    def test_low_quality_is_the_default(self) -> None:
        with (
            patch(
                "room_reconstruction.cli.load_pipeline_config",
                return_value=PipelineConfig(),
            ) as load_config,
            patch("room_reconstruction.cli.run_pipeline"),
        ):
            result = main(["room.mp4", "--output", "results/room"])

        self.assertEqual(result, 0)
        self.assertEqual(load_config.call_args.args[0].name, "low-quality.yml")

    def test_custom_config_cannot_be_combined_with_quality(self) -> None:
        with self.assertRaises(SystemExit):
            main(
                [
                    "room.mp4",
                    "--output",
                    "results/room",
                    "--quality",
                    "low",
                    "--config",
                    str(Path("custom.yml")),
                ]
            )


if __name__ == "__main__":
    unittest.main()
