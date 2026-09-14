import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from room_reconstruction import webapp


class WebAppTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(webapp.app)

    def test_index_explains_local_gpu_workflow(self) -> None:
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("your computer", response.text)

    def test_rejects_unsupported_upload(self) -> None:
        response = self.client.post(
            "/api/jobs",
            files={"video": ("room.txt", b"not a video", "text/plain")},
        )

        self.assertEqual(response.status_code, 400)

    def test_starts_cli_for_supported_upload(self) -> None:
        class FakeProcess:
            def poll(self) -> None:
                return None

        with tempfile.TemporaryDirectory() as directory, patch.object(
            webapp, "RESULTS_ROOT", Path(directory)
        ), patch.object(webapp.subprocess, "Popen", return_value=FakeProcess()) as popen:
            response = self.client.post(
                "/api/jobs",
                files={"video": ("room.mov", b"video bytes", "video/quicktime")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("id", response.json())
        popen.assert_called_once()
