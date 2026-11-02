# sparsekv Port Notes — GPU2 two-stream 262K

Upstream pin: d2d876d2ae4ca6703b04d2b0e2b43a17c4deb634 (main, 2 commits:
c24c4b6 serving release, d2d876d eval/kld).

## Base venv decision (Task 1)

The plan's "venv clone of the serving recipe" resolves to the production
image itself: `hyperqwen-pthfa:032-b` carries `/app/venv` (Python 3.12,
torch 2.13+cu130, vLLM 0.32.0, 65-patch series applied in-image, pthfa
installed, verify.sh --install passed at build). Host lacks python3.12 and
has nvcc 12.4 only, so a host-built venv would diverge from production.

Every port step therefore runs inside a container from
`hyperqwen-pthfa:032-b` (or a `docker commit` variant `:032-sparse-port`),
GPU2-only, volumes from `sparsekv-port/`. The production image is never
modified; production containers `hyperqwen-qwopus-pthfa-gpu{0,2}` are never
stopped before Task 7.

Verified at staging: image reports vllm 0.32.0, patches/series present,
`import pth_flash_attn` OK.

## Fit-corpus provenance (Task 2)

(to be filled)

## Dense-base GPU boot deferral (Task 1 Step 3)

GPU2 carries the production engine (23.1/24 GiB in use); a second full
engine on the same card cannot boot (vLLM refuses at 0.78 GiB free —
expected). The Global Constraint "production is never disrupted before
Task 7" outranks Step 3, so the GPU dense-boot verification is deferred
to the Task 6 swap window (or a Chris-approved maintenance slot).
Interim evidence that the base venv is intact: CPU-mode
`pthfa/tests/gate_test.py` — PASS. Task 6 Step 1 additionally asserts
dense equivalence against the production engine itself.
