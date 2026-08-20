from pathlib import Path
import tempfile
import unittest

from room_reconstruction.pipeline import create_output_layout


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
