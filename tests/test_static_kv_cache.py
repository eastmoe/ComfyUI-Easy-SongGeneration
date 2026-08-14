from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


RUNTIME_ROOT = Path(__file__).resolve().parents[1] / "songgeneration"
if str(RUNTIME_ROOT) not in sys.path:
    sys.path.insert(0, str(RUNTIME_ROOT))

from codeclm.models.llama.configuration_llama import LlamaConfig
from codeclm.models.llama.modeling_llama import LlamaAttention, StaticKVCache


class StaticKVCacheTests(unittest.TestCase):
    def test_static_cache_matches_growing_tuple_cache(self):
        config = LlamaConfig(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=32,
            attention_bias=False,
        )
        torch.manual_seed(7)
        dynamic_attention = LlamaAttention(config).eval()
        static_attention = LlamaAttention(config).eval()
        static_attention.load_state_dict(dynamic_attention.state_dict())
        static_attention.set_static_cache_capacity(8)

        chunks = [
            torch.randn(2, 3, 16),
            torch.randn(2, 1, 16),
            torch.randn(2, 1, 16),
        ]
        dynamic_cache = None
        static_cache = None
        static_key_pointer = None
        position = 0

        with torch.no_grad():
            for chunk in chunks:
                position_ids = torch.arange(position, position + chunk.shape[1]).unsqueeze(0)
                dynamic_output, _, dynamic_cache = dynamic_attention(
                    chunk,
                    position_ids=position_ids,
                    past_key_value=dynamic_cache,
                    use_cache=True,
                )
                static_output, _, static_cache = static_attention(
                    chunk,
                    position_ids=position_ids,
                    past_key_value=static_cache,
                    use_cache=True,
                )
                torch.testing.assert_close(static_output, dynamic_output, rtol=1e-5, atol=1e-6)
                if static_key_pointer is None:
                    static_key_pointer = static_cache.key.data_ptr()
                else:
                    self.assertEqual(static_cache.key.data_ptr(), static_key_pointer)
                position += chunk.shape[1]

        self.assertIsInstance(static_cache, StaticKVCache)
        self.assertEqual(static_cache.length, 5)
        self.assertEqual(static_cache.key.shape[-2], 8)


if __name__ == "__main__":
    unittest.main()
