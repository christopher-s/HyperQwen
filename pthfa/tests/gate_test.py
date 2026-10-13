#!/usr/bin/env python3
"""CPU test of the installed pth_flash_attn package against this venv's vLLM. No GPU, no model.

pthfa/install.sh runs it on the installed copy (pristine vLLM wheel, before patches/), and the Dockerfile
runs it again after patches/apply.sh and kvarn/install.sh (the patched tree the server runs). It checks
what makes the package safe to ship in every image while only PTH_FA=1 uses it:

  1. the op library is inside the installed package, loads without a GPU and registers its op;
  2. exactly one vllm.general_plugins entry point, and calling it the way vLLM does registers the
     backend under AttentionBackendEnum.CUSTOM, idempotently, and never over another plugin's CUSTOM;
  3. inert unless selected: CUSTOM is in no automatic-selection list of this vLLM, for any compute
     capability or KV dtype, so a server without --attention-backend CUSTOM never reaches it;
  4. the backend's declared limits (sm_8x, head sizes 64/128/256, int8_per_token_head, causal
     decoder self-attention, no sliding window, ALiBi, sinks, soft cap);
  5. an explicit selection the kernel cannot serve fails at load with its reason, before anything is
     allocated (sliding window, ALiBi, sinks, soft cap, another KV dtype, another head size);
  6. this vLLM's speculative config accepts "attention_backend":"CUSTOM" and copies it onto the
     drafter's config, which is how PTH_FA=1 puts the MTP drafter on the same backend.

  CUDA_VISIBLE_DEVICES= venv/bin/python pthfa/tests/gate_test.py
"""
import importlib.metadata as metadata
import sys
from pathlib import Path

FAILS = []


def check(cond, what):
    print(f"  {'PASS' if cond else 'FAIL'}  {what}")
    if not cond:
        FAILS.append(what)


def main():
    import torch

    import pth_flash_attn
    import pth_flash_attn.ops as ops

    print("== pthfa gate test")
    pkg = Path(pth_flash_attn.__file__).resolve().parent
    check("site-packages" in pkg.parts, f"pth_flash_attn is the installed copy ({pkg})")
    check(ops.is_available() and Path(ops.LIBRARY_PATH).resolve().parent == pkg,
          f"op library loads without a GPU ({Path(ops.LIBRARY_PATH).name})")
    check(hasattr(torch.ops.pth_flash_attn, "pth_flash_attn_fwd"), "op pth_flash_attn::pth_flash_attn_fwd registered")

    from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

    custom = AttentionBackendEnum.CUSTOM
    ours = "pth_flash_attn.vllm_backend.PthFlashAttnBackend"
    eps = [e for e in metadata.entry_points(group="vllm.general_plugins") if e.name == "pth_flash_attn"]
    check(len(eps) == 1 and eps[0].value == "pth_flash_attn.plugin:register",
          f"one vllm.general_plugins entry point -> {[e.value for e in eps]}")
    register = eps[0].load()
    custom.clear_override()
    register_backend(custom, "some.other.plugin.Backend")
    register()
    check(custom.get_path() == "some.other.plugin.Backend", "a CUSTOM another plugin registered first is left alone")
    custom.clear_override()
    register()
    register()
    check(custom.get_path() == ours, f"the entry point registers CUSTOM -> {ours} (twice: idempotent)")
    backend = custom.get_class()
    check(backend.get_name() == "CUSTOM", "the backend class imports and is named CUSTOM")

    from vllm.platforms.cuda import _get_backend_priorities
    from vllm.platforms.interface import DeviceCapability

    caps = [DeviceCapability(7, 5), DeviceCapability(8, 0), DeviceCapability(8, 6), DeviceCapability(8, 9),
            DeviceCapability(9, 0), DeviceCapability(10, 0), DeviceCapability(12, 0)]
    picked = []
    for cap in caps:
        for kv in (None, "auto", "bfloat16", "fp8", "int8_per_token_head"):
            for hs in (64, 128, 256):
                for non_causal in (False, True):
                    prio = _get_backend_priorities(False, cap, kv_cache_dtype=kv, head_size=hs,
                                                   use_non_causal=non_causal)
                    if custom in prio:
                        picked.append((cap, kv, hs, non_causal))
    check(not picked, f"CUSTOM is in no automatic-selection list ({len(caps) * 5 * 3 * 2} combinations): {picked[:3]}")

    from pth_flash_attn.vllm_backend import PthFlashAttnImpl

    check(list(backend.supported_kv_cache_dtypes) == ["int8_per_token_head"], "KV dtype: int8_per_token_head only")
    check(set(backend.supported_dtypes) == {torch.float16, torch.bfloat16}, "activations: float16 and bfloat16")
    check([h for h in (32, 64, 80, 96, 128, 192, 256, 512) if backend.supports_head_size(h)] == [64, 128, 256],
          "head sizes 64, 128, 256 only")
    ok_caps = [f"{c.major}.{c.minor}" for c in caps if backend.supports_compute_capability(c)]
    check(ok_caps == ["8.0", "8.6", "8.9"], f"compute capability 8.0/8.6/8.9 only -> {ok_caps}")
    check(not any((backend.supports_sliding_window(), backend.supports_sink(), backend.supports_non_causal(),
                   backend.supports_alibi_sqrt(), backend.supports_mm_prefix())),
          "no sliding window, sinks, non-causal, ALiBi or bidirectional mm prefix")
    check(backend.supports_block_size(816) and backend.supports_block_size(16) and not backend.supports_block_size(8),
          "block sizes: multiples of 16 (the hybrid group's 816 included)")

    base = dict(num_heads=24, head_size=256, scale=0.0625, num_kv_heads=4, alibi_slopes=None,
                sliding_window=None, kv_cache_dtype="int8_per_token_head")
    refusals = {
        "sliding window": dict(sliding_window=4096),
        "ALiBi": dict(alibi_slopes=[0.5] * 24),
        "attention sinks": dict(sinks=torch.zeros(24)),
        "logits soft cap": dict(logits_soft_cap=30.0),
        "kv_cache_dtype fp8": dict(kv_cache_dtype="fp8"),
        "kv_cache_dtype auto": dict(kv_cache_dtype="auto"),
        "head size 96": dict(head_size=96),
    }
    for reason, bad in refusals.items():
        try:
            PthFlashAttnImpl(**{**base, **bad})
            got = "constructed"
        except NotImplementedError as e:
            got = str(e)
        check(reason in got, f"explicit selection with {reason} refused at load: {got[:90]}")

    from vllm.config.speculative import _DRAFT_VLLM_CONFIG_OVERRIDES, SpeculativeConfig

    check(SpeculativeConfig._parse_attention_backend("CUSTOM") is custom,
          'speculative config parses "attention_backend":"CUSTOM"')
    check(("attention_backend", "attention_config", "backend") in _DRAFT_VLLM_CONFIG_OVERRIDES,
          "the drafter's config takes the speculative config's attention_backend")

    print(f"pthfa gate test: {'OK' if not FAILS else f'{len(FAILS)} FAILED'}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
