#!/usr/bin/env python3
"""
Quantize SuperQwen3.8-27B-abliterated to AutoRound W4A16 (compressed-tensors)
and apply HyperQwen prepare scripts for RTX 3090 deployment.
"""

import os
import sys
import json
import shutil
import subprocess
from pathlib import Path

# Force GPU 2 (PCIe x16 RTX 3090)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

MODEL_SRC = "/mnt/500gb-ssd/ai/models/SuperQwen3.8-27b-abliterated"
OUTPUT_DIR = "/mnt/500gb-ssd/ai/models/SuperQwen3.8-27B-Uncensored-W4A16-AutoRound"
HQ_ROOT = "/home/chris/tools/hyperqwen"
COMP_DIR = "/home/chris/compressed_models"

print("=== Starting SuperQwen AutoRound W4A16 Quantization ===")
print(f"Source Model: {MODEL_SRC}")
print(f"Target Output: {OUTPUT_DIR}")

# 1. Build layer exclusion config: keep ONLY GDN gates (in_proj_a/b), visual, mtp, and lm_head in fp16/bf16
print("Building layer exclusion config...")
with open(f"{MODEL_SRC}/model.safetensors.index.json") as f:
    idx = json.load(f)

weight_map = idx["weight_map"]
layer_config = {}

for tensor_name in weight_map:
    is_visual = "visual" in tensor_name
    is_mtp = "mtp" in tensor_name
    is_lm_head = "lm_head" in tensor_name
    is_gdn_ab = "linear_attn.in_proj_a" in tensor_name or "linear_attn.in_proj_b" in tensor_name
    
    if is_visual or is_mtp or is_lm_head or is_gdn_ab:
        mod_name = tensor_name
        for suffix in [".weight", ".bias", ".norm.weight", ".norm.bias"]:
            if mod_name.endswith(suffix):
                mod_name = mod_name[:-len(suffix)]
                break
        layer_config[mod_name] = {"bits": 16, "data_type": "float"}

print(f"Layers preserved in float16/bf16: {len(layer_config)}")

from auto_round import AutoRound
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_SRC)

print("Initializing AutoRound compressor (W4A16 group-128)...")
ar = AutoRound(
    model=MODEL_SRC,
    tokenizer=tokenizer,
    scheme="W4A16",
    format="auto_round:llm_compressor",
    layer_config=layer_config,
    nsamples=128,
    seqlen=512,
    device_map=0
)

print("Running AutoRound quantization (this takes ~40 minutes on GPU 2)...")
ar.quantize()

print("AutoRound tuning completed! Assembling target directory...")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 2. Move generated compressed shards from COMP_DIR
comp_actual = COMP_DIR if os.path.exists(COMP_DIR) else "./compressed_models"
for f in os.listdir(comp_actual):
    src = os.path.join(comp_actual, f)
    dst = os.path.join(OUTPUT_DIR, f)
    print(f"Moving {f} to {OUTPUT_DIR}...")
    shutil.move(src, dst)

# 3. Extract MTP tensors from source shard 18 into model_extra_tensors.safetensors
print("Extracting MTP tensors from source shard 18...")
from safetensors.torch import save_file
from safetensors import safe_open

src_shard18 = os.path.join(MODEL_SRC, "model-00018-of-00018.safetensors")
mtp_tensors = {}
with safe_open(src_shard18, framework="pt") as f:
    for k in f.keys():
        if "mtp" in k:
            mtp_tensors[k] = f.get_tensor(k)

print(f"Found {len(mtp_tensors)} MTP tensors")
extra_tensors_path = os.path.join(OUTPUT_DIR, "model_extra_tensors.safetensors")
save_file(mtp_tensors, extra_tensors_path)
print("Saved model_extra_tensors.safetensors")

# Update index.json
idx_path = os.path.join(OUTPUT_DIR, "model.safetensors.index.json")
with open(idx_path) as f:
    out_idx = json.load(f)

for k in mtp_tensors:
    out_idx["weight_map"][k] = "model_extra_tensors.safetensors"

with open(idx_path, "w") as f:
    json.dump(out_idx, f, indent=2)

# 4. Write quantization_config and config.json
ignore_list = sorted(list(layer_config.keys()))
if "lm_head" not in ignore_list:
    ignore_list.append("lm_head")

quant_config = {
    "config_groups": {
        "group_0": {
            "targets": ["Linear"],
            "weights": {
                "num_bits": 4,
                "type": "int",
                "symmetric": True,
                "group_size": 128,
                "strategy": "group",
                "block_structure": None,
                "dynamic": False,
                "actorder": None,
                "scale_dtype": None,
                "zp_dtype": None,
                "observer": "memoryless_minmax",
                "observer_kwargs": {}
            },
            "input_activations": None,
            "output_activations": None,
            "format": "pack-quantized"
        }
    },
    "quant_method": "compressed-tensors",
    "kv_cache_scheme": None,
    "format": "pack-quantized",
    "quantization_status": "compressed",
    "global_compression_ratio": None,
    "ignore": ignore_list
}

with open(f"{OUTPUT_DIR}/quantization_config.json", "w") as f:
    json.dump(quant_config, f, indent=2)

with open(f"{MODEL_SRC}/config.json") as f:
    cfg = json.load(f)

cfg["quantization_config"] = quant_config
with open(f"{OUTPUT_DIR}/config.json", "w") as f:
    json.dump(cfg, f, indent=2)

# Copy all metadata and tokenizer files
for f in os.listdir(MODEL_SRC):
    if f.endswith((".json", ".jinja", ".txt", ".md")) and not f.startswith("."):
        dst = os.path.join(OUTPUT_DIR, f)
        if not os.path.exists(dst):
            shutil.copy2(os.path.join(MODEL_SRC, f), dst)
            print(f"Copied {f}")

print("=== Base W4A16 Quantization Complete ===")

# 5. Run HyperQwen preparation scripts (heads & MTP drafting)
print("=== Running HyperQwen Prepare Scripts ===")
hq_python = sys.executable

commands = [
    [hq_python, f"{HQ_ROOT}/prepare/quant_lm_head.py", OUTPUT_DIR],
    [hq_python, f"{HQ_ROOT}/prepare/quant_embed.py", OUTPUT_DIR],
    [hq_python, f"{HQ_ROOT}/prepare/quant_mtp.py", OUTPUT_DIR],
    [hq_python, f"{HQ_ROOT}/prepare/build_draft_vocab.py", OUTPUT_DIR, "--ids", f"{HQ_ROOT}/prepare/draft_vocab_ids.json"]
]

for cmd in commands:
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# Clean up .bak files to preserve disk space
for p in Path(OUTPUT_DIR).glob("*.bak*"):
    p.unlink()

print("=== All Prepare Steps Complete! SuperQwen W4A16 is ready for HyperQwen. ===")
