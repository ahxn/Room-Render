import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from room_reconstruction.errors import ExportError
from room_reconstruction.export import export_gaussian_splat


class ExportTests(unittest.TestCase):
    def test_export_gaussian_splat_records_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            run_dir = root / "reconstruction" / "run"
            models = run_dir / "nerfstudio_models"
            models.mkdir(parents=True)
            config_path = run_dir / "config.yml"
            config_path.write_text("config", encoding="utf-8")
            (models / "step-0001.ckpt").write_bytes(b"checkpoint")

            def create_export(command: list[str], **_: object) -> None:
                destination = Path(command[command.index("--output-dir") + 1])
                filename = command[command.index("--output-filename") + 1]
                (destination / filename).write_bytes(b"ply")

            with patch("room_reconstruction.export.run_command", side_effect=create_export) as command:
                exported = export_gaussian_splat(root)

            self.assertEqual(exported, root / "exports" / "splat.ply")
            self.assertEqual(command.call_args.args[0][0:2], ["ns-export", "gaussian-splat"])
            metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["exports"][0]["path"], str(exported.resolve()))
            self.assertEqual(metadata["exports"][0]["size_bytes"], 3)

    def test_export_rejects_path_as_filename(self) -> None:
        with (
            tempfile.TemporaryDirectory() as temporary_directory,
            self.assertRaises(ExportError),
        ):
            export_gaussian_splat(Path(temporary_directory), filename="nested/splat.ply")
