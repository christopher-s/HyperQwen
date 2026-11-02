#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 sparsekv-hyperqwen contributors
# Install (or remove) the sparse-KV release into an existing HyperQwen venv.
#
#   bash sparsekv/install.sh VENV [--basis FILE]   install / upgrade (idempotent)
#   bash sparsekv/install.sh VENV --check          report what is installed, change nothing
#   bash sparsekv/install.sh VENV --uninstall      reverse everything this script did
#
# VENV must already carry the HyperQwen series (bash patches/apply.sh <venv>/.../vllm) and the
# stock pth_flash_attn built from this repo. What install does:
#   1. vLLM:  bash patches/apply.sh --sparse <site-packages>/vllm   (the sparse-*.patch entries
#      of patches/series; each one is inert unless its SPARSE_* knob is set)
#   2. pthfa: sparsekv/patches/pthfa-sparse-kv.patch applied -p1 to <site-packages>/pth_flash_attn
#      (adds sparse_kv.py and the SPARSE_KV hook in vllm_backend.py; inert unless SPARSE_KV=1).
#      The same change is already in this repo's pthfa/python sources, so a pthfa rebuild from
#      this repo carries it too.
#   3. module: the runtime files of sparsekv/ (ref, triton_ops, tier, tier_connector) into
#      <site-packages>/sparsekv, so the server and the KV connector import them from any cwd.
#   4. basis: --basis FILE (default sparsekv/basis/basis27.pt) copied to
#      <VENV>/share/sparsekv/basis27.pt (+ .sha256). serve-sparse.sh defaults SPARSE_BASIS to it.
#   5. stamp: <site-packages>/sparsekv/RELEASE (git commit, basis sha256, date); copies of the applied patches in
#      <VENV>/share/sparsekv/applied/, so a later install of a newer release reverses them first (upgrade) and
#      --uninstall reverses exactly what was applied.
#
# Exit codes: 0 ok; 1 a step failed (message names it); 2 usage error.
set -euo pipefail
HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd -- "$HERE/.." && pwd)
RUNTIME_FILES=(__init__.py ref.py triton_ops.py tier.py tier_connector.py)
PTHFA_PATCH="$HERE/patches/pthfa-sparse-kv.patch"

usage() { sed -n '2,6p' "$0" >&2; exit 2; }
[ $# -ge 1 ] || usage
VENV=$(cd -- "$1" && pwd) || usage
shift
MODE=install
BASIS_SRC=$HERE/basis/basis27.pt
while [ $# -gt 0 ]; do
  case "$1" in
    --check) MODE=check ;;
    --uninstall) MODE=uninstall ;;
    --basis) BASIS_SRC=$2; shift ;;
    *) usage ;;
  esac
  shift
done

PY="$VENV/bin/python"
[ -x "$PY" ] || { echo "ERROR: $PY not found" >&2; exit 1; }
SP=$("$PY" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
VLLM_DIR="$SP/vllm"
PTHFA_DIR="$SP/pth_flash_attn"
MOD_DIR="$SP/sparsekv"
BASIS_DST="$VENV/share/sparsekv/basis27.pt"
APPLIED="$VENV/share/sparsekv/applied"   # copies of the patches as applied, so upgrades/uninstall can reverse them
[ -d "$VLLM_DIR" ] || { echo "ERROR: no vllm package in $SP" >&2; exit 1; }
[ -d "$PTHFA_DIR" ] || { echo "ERROR: no pth_flash_attn package in $SP" >&2; exit 1; }

# 0 = applied, 1 = not applied (forward applies cleanly), 2 = neither (conflict / partial)
patch_state() {  # DIR PATCH
  if patch -p1 -R --dry-run -s --fuzz 0 -d "$1" -i "$2" >/dev/null 2>&1; then return 0; fi
  if patch -p1 --forward --dry-run -s --fuzz 0 -d "$1" -i "$2" >/dev/null 2>&1; then return 1; fi
  return 2
}

sparse_vllm_patches() { bash "$REPO/patches/apply.sh" --list | grep '^sparse-.*\.patch$'; }

report() {
  local rc=0 st name
  for name in $(sparse_vllm_patches); do
    st=0; patch_state "$VLLM_DIR" "$REPO/patches/$name" || st=$?
    case $st in 0) echo "vllm   $name: applied" ;; 1) echo "vllm   $name: NOT applied"; rc=1 ;;
                *) echo "vllm   $name: CONFLICT (partly applied or a different tree)"; rc=1 ;; esac
  done
  st=0; patch_state "$PTHFA_DIR" "$PTHFA_PATCH" || st=$?
  case $st in 0) echo "pthfa  pthfa-sparse-kv.patch: applied" ;; 1) echo "pthfa  pthfa-sparse-kv.patch: NOT applied"; rc=1 ;;
              *) echo "pthfa  pthfa-sparse-kv.patch: CONFLICT"; rc=1 ;; esac
  local f
  for f in "${RUNTIME_FILES[@]}"; do
    if cmp -s "$HERE/$f" "$MOD_DIR/$f"; then echo "module sparsekv/$f: matches repo"
    else echo "module sparsekv/$f: MISSING or differs from repo"; rc=1; fi
  done
  if [ -f "$BASIS_DST" ] && (cd "$(dirname "$BASIS_DST")" && sha256sum -c --quiet basis27.pt.sha256) ; then
    echo "basis  $BASIS_DST: ok ($(cut -c1-16 "$BASIS_DST.sha256")...)"
  else echo "basis  $BASIS_DST: MISSING or checksum mismatch"; rc=1; fi
  [ -f "$MOD_DIR/RELEASE" ] && sed 's/^/stamp  /' "$MOD_DIR/RELEASE"
  return $rc
}

case $MODE in
check)
  report; exit $? ;;

uninstall)
  [ -f "$APPLIED/pthfa/pthfa-sparse-kv.patch" ] && PTHFA_PATCH="$APPLIED/pthfa/pthfa-sparse-kv.patch"
  st=0; patch_state "$PTHFA_DIR" "$PTHFA_PATCH" || st=$?
  if [ $st = 0 ]; then
    patch -p1 -R --fuzz 0 --no-backup-if-mismatch -s -d "$PTHFA_DIR" -i "$PTHFA_PATCH"
    echo "== reversed pthfa-sparse-kv.patch"
  else echo "== pthfa-sparse-kv.patch not applied (state $st), left alone"; fi
  for name in $(sparse_vllm_patches | tac); do
    pf="$REPO/patches/$name"; [ -f "$APPLIED/vllm/$name" ] && pf="$APPLIED/vllm/$name"
    st=0; patch_state "$VLLM_DIR" "$pf" || st=$?
    if [ $st = 0 ]; then
      patch -p1 -R --fuzz 0 --no-backup-if-mismatch -s -d "$VLLM_DIR" -i "$pf"
      echo "== reversed $name"
    else echo "== $name not applied (state $st), left alone"; fi
  done
  rm -rf "$MOD_DIR" "$VENV/share/sparsekv" "$PTHFA_DIR/__pycache__/sparse_kv."*
  echo "== removed $MOD_DIR and $VENV/share/sparsekv"
  exit 0 ;;
esac

# ---- install ----
echo "[install] venv $VENV"
# Upgrade: a previously applied patch whose content changed is reversed first (from its stored copy).
for old in "$APPLIED"/vllm/*.patch "$APPLIED"/pthfa/*.patch; do
  [ -f "$old" ] || continue
  name=$(basename "$old"); kind=$(basename "$(dirname "$old")")
  if [ "$kind" = vllm ]; then dir=$VLLM_DIR; cur=$REPO/patches/$name; else dir=$PTHFA_DIR; cur=$PTHFA_PATCH; fi
  cmp -s "$old" "$cur" && continue
  st=0; patch_state "$dir" "$old" || st=$?
  case $st in
    0) patch -p1 -R --fuzz 0 --no-backup-if-mismatch -s -d "$dir" -i "$old"
       echo "== upgrade: reversed the previously installed $name" ;;
    1) echo "== upgrade: previously installed $name is no longer applied; dropping its stored copy" ;;
    *) # G-24: keep the stored copy so --uninstall can still reverse it exactly; do not apply over a mixed tree
       echo "FAILED: upgrade: the previously installed $name neither reverses nor is absent in $dir" >&2
       echo "  (tree modified or partly applied). Stored copy kept at $old; fix the tree, then rerun." >&2
       exit 1 ;;
  esac
  rm -f "$old"
done
echo "[install] 1/5 vLLM sparse patches -> $VLLM_DIR"
bash "$REPO/patches/apply.sh" --sparse "$VLLM_DIR"

echo "[install] 2/5 pthfa patch -> $PTHFA_DIR"
st=0; patch_state "$PTHFA_DIR" "$PTHFA_PATCH" || st=$?
case $st in
  0) echo "== pthfa-sparse-kv.patch (already applied)" ;;
  1) patch -p1 --forward --fuzz 0 --no-backup-if-mismatch -s -d "$PTHFA_DIR" -i "$PTHFA_PATCH"
     echo "== pthfa-sparse-kv.patch" ;;
  *) echo "FAILED: pthfa-sparse-kv.patch neither applies nor is applied in $PTHFA_DIR." >&2
     echo "  The installed pth_flash_attn is not the stock build of this repo's base (or an older" >&2
     echo "  sparse snapshot, e.g. sk-backend's patch, is installed). Restore stock vllm_backend.py and" >&2
     echo "  remove sparse_kv.py (or reinstall pthfa), then rerun." >&2
     exit 1 ;;
esac

mkdir -p "$APPLIED/vllm" "$APPLIED/pthfa"
for name in $(sparse_vllm_patches); do install -m 0644 "$REPO/patches/$name" "$APPLIED/vllm/$name"; done
install -m 0644 "$PTHFA_PATCH" "$APPLIED/pthfa/pthfa-sparse-kv.patch"

echo "[install] 3/5 sparsekv module -> $MOD_DIR"
mkdir -p "$MOD_DIR"
for f in "${RUNTIME_FILES[@]}"; do install -m 0644 "$HERE/$f" "$MOD_DIR/$f"; done
rm -rf "$MOD_DIR/__pycache__"

echo "[install] 4/5 basis $BASIS_SRC -> $BASIS_DST"
[ -f "$BASIS_SRC" ] || { echo "FAILED: basis file $BASIS_SRC not found" >&2; exit 1; }
mkdir -p "$(dirname "$BASIS_DST")"
install -m 0644 "$BASIS_SRC" "$BASIS_DST"
(cd "$(dirname "$BASIS_DST")" && sha256sum basis27.pt > basis27.pt.sha256)
"$PY" - "$BASIS_DST" <<'EOF'
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu")
L, b = list(d["layers"]), d["basis"]
assert L == list(range(3, 64, 4)), L
assert tuple(b.shape) == (16, 4, 64, 256), tuple(b.shape)
print(f"== basis ok: 16 layers, shape {tuple(b.shape)}, r={d.get('r')}")
EOF

echo "[install] 5/5 stamp + import check"
{
  echo "commit $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)$(git -C "$REPO" diff --quiet HEAD -- sparsekv patches pthfa 2>/dev/null || echo ' (+uncommitted changes)')"
  echo "basis_sha256 $(cut -d' ' -f1 "$BASIS_DST.sha256")"
  echo "installed $(date -u +%FT%TZ) from $REPO"
} > "$MOD_DIR/RELEASE"
(cd / && "$PY" -c '
import pth_flash_attn.sparse_kv, pth_flash_attn.vllm_backend as vb, sparsekv.triton_ops, sparsekv.tier, sparsekv.tier_connector as tc
assert hasattr(vb, "sparse_kv"), "vllm_backend has no sparse hook"
assert hasattr(tc, "HostTierConnector"), "tier_connector has no HostTierConnector (vLLM import failed?)"
print("== imports ok:", sparsekv.__file__)
' 2>&1 | grep -v -i warn)
report >/dev/null || { report; echo "FAILED: post-install check" >&2; exit 1; }
echo "[install] done"
