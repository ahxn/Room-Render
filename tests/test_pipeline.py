import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from room_reconstruction.config import PipelineConfig, VideoConfig
from room_reconstruction.errors import OutputExistsError
from room_reconstruction.pipeline import create_output_layout, find_training_config, run_pipeline
from room_reconstruction.video import VideoMetadata


class PipelineTests(unittest.TestCase):
    def test_create_output_layout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = create_output_layout(Path(temporary_directory) / "result")
            self.assertTrue(paths.root.is_dir())
            self.assertTrue(paths.logs.is_dir())
            self.assertTrue(paths.frames.is_dir())
            self.assertTrue(paths.processed.is_dir())
            self.assertTrue(paths.reconstruction.is_dir())
            self.assertEqual(paths.metadata, paths.root / "metadata.json")

    def test_find_training_config_requires_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            incomplete = root / "older"
            incomplete.mkdir()
            (incomplete / "config.yml").write_text("incomplete", encoding="utf-8")

            complete = root / "newer"
            models = complete / "nerfstudio_models"
            models.mkdir(parents=True)
            config_path = complete / "config.yml"
            config_path.write_text("complete", encoding="utf-8")
            (models / "step-0001.ckpt").write_bytes(b"checkpoint")

            self.assertEqual(find_training_config(root), config_path)

    def test_fresh_run_rejects_existing_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "room.mp4"
            video_path.write_bytes(b"video")
            output = root / "output"
            paths = create_output_layout(output)
            (paths.frames / "frame_00001.jpg").write_bytes(b"frame")

            with (
                patch("room_reconstruction.pipeline.validate_video", return_value=_video()),
                self.assertRaises(OutputExistsError),
            ):
                run_pipeline(video_path, output)

    def test_resume_reuses_completed_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "room.mp4"
            video_path.write_bytes(b"video")
            output = root / "output"
            paths = create_output_layout(output)
            (paths.frames / "frame_00001.jpg").write_bytes(b"frame")
            (paths.processed / "transforms.json").write_text(
                json.dumps({"frames": [{}]}), encoding="utf-8"
            )
            run_dir = paths.reconstruction / "run"
            models = run_dir / "nerfstudio_models"
            models.mkdir(parents=True)
            config_path = run_dir / "config.yml"
            config_path.write_text("config", encoding="utf-8")
            (models / "step-0001.ckpt").write_bytes(b"checkpoint")
            config = PipelineConfig(video=VideoConfig(target_frame_count=1))

            with (
                patch("room_reconstruction.pipeline.validate_video", return_value=_video()),
                patch("room_reconstruction.pipeline.run_command") as run_command,
            ):
                result = run_pipeline(video_path, output, config=config, resume=True)

            run_command.assert_not_called()
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["result"]["config_path"], str(config_path))
            self.assertEqual(result["stages"]["frames"]["status"], "reused")
            self.assertEqual(result["stages"]["camera_poses"]["status"], "reused")
            self.assertEqual(result["stages"]["training"]["status"], "reused")

    def test_dry_run_plans_safe_overwrite_and_viewer_shutdown(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            video_path = root / "room.mp4"
            video_path.write_bytes(b"video")
            output = root / "output"

            with (
                patch("room_reconstruction.pipeline.validate_video", return_value=_video()),
                patch("room_reconstruction.pipeline.run_command") as run_command,
            ):
                result = run_pipeline(video_path, output, dry_run=True)

            commands = [call.args[0] for call in run_command.call_args_list]
            self.assertIn("-y", commands[0])
            self.assertEqual(commands[-1][-2:], ["--viewer.quit-on-train-completion", "True"])
            self.assertEqual(result["status"], "dry_run")


def _video() -> VideoMetadata:
    return VideoMetadata(
        duration_seconds=10.0,
        width=1920,
        height=1080,
        frame_rate=30.0,
        file_size_bytes=100,
        codec="h264",
    )
