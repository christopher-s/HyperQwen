# In-container patches (sparsekv port)

Applied to /app/single-user/start_qwen.sh inside hyperqwen-pthfa:032-sparse-port
containers (docker cp + docker exec). NOT in the repo's start_qwen.sh.

1. apply-in-container-guard-bypass.py - the PTH_FA=1 + --kv-transfer-config
   refusal arm becomes *__never_match_sparsekv_port* (dead). Rationale: the
   generic CPU-tier guard predates the sparsekv port; HostTierConnector +
   PTH_FA is upstream's validated combination.
2. apply-in-container-alloc-arm.py - the ALLOC_DEFAULT arm (expandable_segments
   -> False when a KV connector is present) keeps its real pattern
   *" --kv-transfer-config"*; vLLM refuses the connector under VMM.

A future proper build bakes both into the variant image's start_qwen.sh
(docker commit after applying) so boots need no in-container patching.
