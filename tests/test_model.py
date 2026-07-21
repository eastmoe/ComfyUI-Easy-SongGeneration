from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.easy_songgeneration.model import SongGenerationModelHandle


class OptionalAutoPromptTests(unittest.TestCase):
    def test_missing_weights_only_fail_when_auto_style_is_selected(self):
        handle = SongGenerationModelHandle.__new__(SongGenerationModelHandle)
        handle.auto_prompt = None
        options = SimpleNamespace(
            prompt_audio=None,
            auto_prompt_audio_type="Pop",
        )

        with self.assertRaisesRegex(RuntimeError, "Automatic style prompt weights are unavailable"):
            handle._prepare_conditioning(options)

    def test_missing_weights_allow_plain_text_conditioning(self):
        handle = SongGenerationModelHandle.__new__(SongGenerationModelHandle)
        handle.auto_prompt = None
        options = SimpleNamespace(
            prompt_audio=None,
            auto_prompt_audio_type="None",
        )

        result = handle._prepare_conditioning(options)

        self.assertTrue(result["melody_is_wav"])
        self.assertIsNone(result["pmt_wav"])


if __name__ == "__main__":
    unittest.main()
