# SPDX-License-Identifier: Apache-2.0
"""Port-specific test: the tier must refuse to pin host RAM that would leave
less than the floor of free memory (plan Task 3 Step 2). No GPU needed.

CUDA_VISIBLE_DEVICES= python sparsekv/test_port_ram_guard.py
"""
import os
import sys
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "pthfa", "python"))

import torch  # noqa: E402


def _read_avail_bytes():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("MemAvailable not found")


class TestRamHeadroomGuard(unittest.TestCase):
    def test_helper_exists_and_refuses_when_over_floor(self):
        from sparsekv.tier import _assert_ram_headroom

        avail = _read_avail_bytes()
        huge = avail  # asking for ALL available memory leaves 0 < floor
        with self.assertRaises(RuntimeError) as cm:
            _assert_ram_headroom(huge, floor_gb=8.0)
        self.assertIn("refusing to pin", str(cm.exception))

    def test_allows_reasonable_pin(self):
        from sparsekv.tier import _assert_ram_headroom

        avail = _read_avail_bytes()
        small = 1 << 20
        if (avail - small) < int(8.0 * (1 << 30)):
            self.skipTest("host has <8 GiB free; cannot test the allow path")
        _assert_ram_headroom(small, floor_gb=8.0)  # must not raise


class TestAllocPinnedEnforcement(unittest.TestCase):
    def test_alloc_pinned_refuses_oversized(self):
        from sparsekv.tier import alloc_pinned

        avail = _read_avail_bytes()
        huge_tokens = (avail // 4) + (1 << 30)  # more than ALL available, in int32 tokens
        with self.assertRaises(RuntimeError) as cm:
            alloc_pinned((huge_tokens,), dtype=torch.int32)
        self.assertIn("refusing to pin", str(cm.exception))

    def test_alloc_pinned_opt_out_flag(self):
        from sparsekv import tier as tier_mod
        from sparsekv.tier import alloc_pinned

        # opt-out path must skip the guard entirely. Spy proves the guard never
        # runs with headroom_floor_gb=None, even for an over-budget request.
        huge_tokens = (_read_avail_bytes() // 4) + (1 << 30)
        called = []
        real = tier_mod._assert_ram_headroom

        def spy(nbytes, floor_gb=8.0):
            called.append((nbytes, floor_gb))
            return real(nbytes, floor_gb)

        with mock.patch.object(tier_mod, "_assert_ram_headroom", spy):
            try:
                alloc_pinned((huge_tokens,), dtype=torch.int32, headroom_floor_gb=None)
                # on a GPU host with this much RAM this could legitimately succeed
            except RuntimeError as e:
                # CPU-only container: cudaHostRegister fails with a driver error;
                # the guard's message must be absent either way
                self.assertNotIn("refusing to pin", str(e))
        self.assertEqual(called, [], "guard ran despite headroom_floor_gb=None")



if __name__ == "__main__":
    unittest.main(verbosity=2)

