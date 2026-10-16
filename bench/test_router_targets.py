#!/usr/bin/env python3
"""CPU-only route contract for Black Monolith's twin SuperQwen engines."""
import importlib.util
from pathlib import Path
import unittest

REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("hyperqwen_router", REPO / "hyperqwen_router.py")
assert SPEC is not None and SPEC.loader is not None
ROUTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ROUTER)


class RouterTargetTests(unittest.TestCase):
    def test_canonical_aliases_target_current_superqwen_ports(self):
        self.assertEqual(ROUTER.target_for_model("qwen-gpu0"), ROUTER.PORT_GPU0)
        self.assertEqual(ROUTER.target_for_model("black-monolith/qwen-gpu0"), ROUTER.PORT_GPU0)
        self.assertEqual(ROUTER.target_for_model("qwen-gpu2"), ROUTER.PORT_GPU2)
        self.assertEqual(ROUTER.target_for_model("black-monolith/qwen-gpu2"), ROUTER.PORT_GPU2)

    def test_fast_alias_targets_gpu2_and_default_targets_gpu0(self):
        self.assertEqual(ROUTER.target_for_model("qwen-3.8-27b-fast"), ROUTER.PORT_GPU2)
        self.assertEqual(ROUTER.target_for_model("qwen-3.8-27b"), ROUTER.PORT_GPU0)

    def test_current_engines_never_fall_back_to_retired_alt_ports(self):
        self.assertIsNone(ROUTER.fallback_for_target(ROUTER.PORT_GPU0))
        self.assertIsNone(ROUTER.fallback_for_target(ROUTER.PORT_GPU2))


if __name__ == "__main__":
    unittest.main()
