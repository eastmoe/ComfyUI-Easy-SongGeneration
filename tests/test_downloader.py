from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.easy_songgeneration import downloader


class AutoPromptDownloadTests(unittest.TestCase):
    def test_selected_mirror_does_not_fall_back_to_official_endpoint(self):
        calls = []

        def fail_download(**kwargs):
            calls.append(kwargs["endpoint"])
            raise TimeoutError("mirror redirected to an unavailable host")

        with tempfile.TemporaryDirectory() as directory, patch.object(
            downloader, "_hf_hub_download", side_effect=fail_download
        ):
            with self.assertRaisesRegex(RuntimeError, "Failed to download"):
                downloader._ensure_auto_prompt_asset(
                    target_root=Path(directory),
                    endpoint="https://hf-mirror.com",
                    overwrite_existing=False,
                )

        self.assertEqual(calls, ["https://hf-mirror.com"])

    def test_optional_auto_prompt_failure_does_not_fail_model_download(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(
            downloader, "_songgen_model_root", return_value=Path(directory)
        ), patch.object(downloader, "_dry_run_supported", return_value=False), patch.object(
            downloader, "_snapshot_download", return_value=directory
        ), patch.object(
            downloader, "_ensure_auto_prompt_asset", side_effect=RuntimeError("Space unavailable")
        ), patch.object(
            downloader, "_verify_selected_local_files", return_value={"checked": 0}
        ):
            result = json.loads(
                downloader.download_songgeneration_assets(
                    source="hf-mirror.com",
                    model_choice="runtime-only",
                    revision="main",
                    overwrite_existing=False,
                )
            )

        self.assertEqual(result["auto_prompt"]["status"], "unavailable")
        self.assertEqual(result["auto_prompt_integrity_check"], {"status": "unavailable"})


if __name__ == "__main__":
    unittest.main()
