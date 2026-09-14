import json
import unittest

from room_reconstruction.errors import InputValidationError
from room_reconstruction.video import (
    parse_ffprobe_output,
    parse_frame_rate,
    resolution_meets_minimum,
    sampling_rate,
    source_quality_advisories,
)


class VideoTests(unittest.TestCase):
    def test_parse_frame_rate_fraction(self) -> None:
        self.assertAlmostEqual(parse_frame_rate("30000/1001"), 29.97002997)


    def test_parse_ffprobe_output_uses_video_stream(self) -> None:
        payload = json.dumps(
            {
                "streams": [
                    {"codec_type": "audio", "codec_name": "aac"},
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "avg_frame_rate": "30/1",
                    },
                ],
                "format": {"duration": "43.8"},
            }
        )
        result = parse_ffprobe_output(payload, file_size_bytes=1234)
        self.assertEqual(result.duration_seconds, 43.8)
        self.assertEqual(result.width, 1920)
        self.assertEqual(result.height, 1080)
        self.assertEqual(result.frame_rate, 30.0)
        self.assertEqual(result.file_size_bytes, 1234)
        self.assertEqual(result.codec, "h264")
        self.assertIsNone(result.bit_rate_bits_per_second)


    def test_parse_ffprobe_output_rejects_missing_video(self) -> None:
        with self.assertRaisesRegex(InputValidationError, "valid video stream"):
            parse_ffprobe_output('{"streams": []}', file_size_bytes=0)


    def test_sampling_rate_targets_requested_frame_count(self) -> None:
        self.assertEqual(sampling_rate(50.0, 250), 5.0)


    def test_resolution_accepts_portrait_orientation(self) -> None:
        self.assertTrue(resolution_meets_minimum(1080, 1920, 1280, 720))


    def test_resolution_rejects_insufficient_short_side(self) -> None:
        self.assertFalse(resolution_meets_minimum(640, 1920, 1280, 720))

    def test_source_quality_advisory_flags_low_bitrate_hd_video(self) -> None:
        payload = json.dumps(
            {
                "streams": [
                    {
                        "codec_type": "video",
                        "codec_name": "h264",
                        "width": 1920,
                        "height": 1080,
                        "avg_frame_rate": "30/1",
                    }
                ],
                "format": {"duration": "43.8", "bit_rate": "4300000"},
            }
        )
        metadata = parse_ffprobe_output(payload, file_size_bytes=1234)

        advisories = source_quality_advisories(metadata)

        self.assertEqual(len(advisories), 1)
        self.assertIn("compressed copy", advisories[0])
