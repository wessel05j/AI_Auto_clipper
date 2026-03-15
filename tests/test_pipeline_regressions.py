from __future__ import annotations

import logging
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from core.clipping import (
    _copy_trim_start_is_accurate,
    _extract_clip_stream_copy,
)


class ClipExtractionRegressionTests(unittest.TestCase):
    def test_copy_trim_accuracy_requires_nearby_keyframe(self) -> None:
        self.assertEqual(_copy_trim_start_is_accurate(10.0, 9.8), (True, ""))
        accurate, reason = _copy_trim_start_is_accurate(10.0, 8.5)
        self.assertFalse(accurate)
        self.assertIn("nearest source keyframe", reason)

    @patch("core.clipping._validate_media_output")
    @patch("core.clipping._verify_copy_trim_accuracy")
    @patch("core.clipping._run_ffmpeg_command")
    def test_extract_clip_falls_back_to_accurate_reencode_when_copy_trim_is_rejected(
        self,
        run_ffmpeg: Mock,
        verify_copy_trim_accuracy: Mock,
        validate_media_output: Mock,
    ) -> None:
        run_ffmpeg.side_effect = [(True, ""), (True, ""), (True, "")]
        verify_copy_trim_accuracy.side_effect = [
            (False, "nearest source keyframe is 2.000s before requested start"),
            (False, "nearest source keyframe is 2.000s before requested start"),
        ]
        validate_media_output.side_effect = [
            (True, "", 10.0),
            (True, "", 10.0),
            (True, "", 10.0),
        ]
        logger = logging.getLogger("test.clip_fallback")
        logger.handlers = []
        logger.addHandler(logging.NullHandler())
        logger.propagate = False
        logger.setLevel(logging.CRITICAL)

        output_path = _extract_clip_stream_copy(
            ffmpeg_bin="ffmpeg",
            ffprobe_bin="ffprobe",
            source_video=Path("source.mp4"),
            mp4_output=Path("clip.mp4"),
            mkv_output=Path("clip.mkv"),
            start=10.0,
            duration=10.0,
            logger=logger,
        )

        self.assertEqual(output_path, Path("clip.mp4"))
        self.assertEqual(run_ffmpeg.call_count, 3)
        reencode_command = run_ffmpeg.call_args_list[-1].args[0]
        self.assertIn("libx264", reencode_command)
        self.assertIn("aac", reencode_command)


if __name__ == "__main__":
    unittest.main()
