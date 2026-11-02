"""KLD recorder: a Model Runner V2 custom logits processor for teacher-forced KL-divergence tests.

Per request (opted in via SamplingParams.extra_args / the OpenAI `vllm_xargs` field):
  kld_id    sample id (opt-in marker)
  kld_mode  "ref" (free greedy generation, the reference) or any label for a forced run
  kld_plen  prompt length in tokens (output step t = seq_len - plen)
  kld_out   path of the .npz written when the last step has been recorded
  kld_ref   (forced runs) path of the reference .npz: its tokens are forced, and this run's
            log-probabilities are gathered at the reference's top-K token ids
  kld_eos   list of EOS ids (ref runs stop recording when greedy picks one)
  kld_switch "dense" / "sparse": on a SPARSE_KV=1 server started with KLD_TOGGLE=1, sets the
            per-step dense/sparse switch for this request (sparse_kv cutoff table)

At every output step the processor records, BEFORE forcing: the full-vocab log-softmax of the
model's logits, reduced to (a) the top-K ids/log-probs of this run and (b) for forced runs this
run's log-probs at the reference's top-K ids. Then, for forced runs, it masks the logits so the
sampler picks the reference token. It also records the token that was fed at that step (to prove
the forced path matched the reference) and whether the step's attention ran the sparse decode
path (pth_flash_attn.sparse_kv.step_sparse()).
Speculative decoding must be off (vLLM rejects custom logits processors with it).
"""
import json
import os

import numpy as np
import torch

from vllm.v1.worker.gpu.sample.logits_processor.interface import LogitsProcessor

K_REF = int(os.environ.get("KLD_TOPK", "1024"))   # top-K kept for the reference distribution
K_OWN = int(os.environ.get("KLD_TOPK_OWN", "64"))  # top-K kept for forced runs (diagnostics)


def _skv():
    try:
        import pth_flash_attn.sparse_kv as skv
        return skv
    except Exception:
        return None


class KLDRecorder(LogitsProcessor):
    def __init__(self, vllm_config, req_states):
        self.dev = req_states.device
        self.slots = {}
        self.vcfg = vllm_config
        self.out_root = os.environ.get("KLD_OUT", "")
        self.toggle = os.environ.get("KLD_TOGGLE", "0") == "1"
        self.info_done = False

    @classmethod
    def validate_params(cls, sampling_params):
        return None

    def _info(self):
        if self.info_done or not self.out_root:
            return
        self.info_done = True
        cc = self.vcfg.cache_config
        skv = _skv()
        info = {"block_size": cc.block_size, "num_gpu_blocks": getattr(cc, "num_gpu_blocks", None),
                "speculative": self.vcfg.speculative_config is not None,
                "sparse_kv": bool(skv and skv.enabled()), "toggle": self.toggle,
                "max_model_len": self.vcfg.model_config.max_model_len}
        os.makedirs(self.out_root, exist_ok=True)
        with open(os.path.join(self.out_root, "server_info.json"), "w") as f:
            json.dump(info, f)

    def add_request(self, req_idx, sampling_params):
        xa = sampling_params.extra_args or {}
        sid = xa.get("kld_id")
        if not sid:
            self.slots.pop(req_idx, None)
            return False
        self._info()
        st = {"sid": sid, "mode": xa.get("kld_mode", "ref"), "plen": int(xa["kld_plen"]),
              "n": int(sampling_params.max_tokens), "out": xa["kld_out"],
              "eos": set(int(e) for e in xa.get("kld_eos", [])), "recs": {},
              "forced": None, "refids": None}
        ref = xa.get("kld_ref")
        if ref:
            z = np.load(ref)
            st["forced"] = [int(t) for t in z["token"]]
            st["refids"] = torch.from_numpy(z["top_id"].astype(np.int64)).to(self.dev)
            st["n"] = min(st["n"], len(st["forced"]))
        sw = xa.get("kld_switch")
        skv = _skv()
        if sw and self.toggle and skv is not None and skv.enabled():
            # Non-empty tables keep both FULL graph families in use; {1: 1} = always sparse,
            # {1: 10**9} = always dense. decide() reads this every step.
            skv._CUTOFFS = {1: 1} if sw == "sparse" else {1: 10 ** 9}
        self.slots[req_idx] = st
        return True

    def apply(self, logits, ctx):
        if not self.slots:
            return logits
        idx = ctx.idx_mapping_np
        sl = ctx.seq_lens_upper_bound_np
        skv = _skv()
        sparse_on = bool(skv and skv.enabled())
        for i in range(len(idx)):
            st = self.slots.get(int(idx[i]))
            if st is None:
                continue
            t = int(sl[i]) - st["plen"]
            if t < 0 or t >= st["n"]:
                continue
            lp = torch.log_softmax(logits[i].float(), dim=-1)
            k = K_REF if st["forced"] is None else K_OWN
            top = torch.topk(lp, k)
            rec = {"top_lp": top.values.cpu().numpy(), "top_id": top.indices.to(torch.int32).cpu().numpy(),
                   "inp": int(ctx.input_ids[i]), "sparse": bool(sparse_on and skv.step_sparse())}
            if st["refids"] is not None:
                rec["q_ref"] = lp[st["refids"][t]].cpu().numpy()
            st["recs"][t] = rec
            last = t == st["n"] - 1
            if st["forced"] is not None:
                tok = st["forced"][t]
                logits[i].fill_(float("-inf"))
                logits[i, tok] = 0.0
            elif int(rec["top_id"][0]) in st["eos"]:
                last = True
            if last:
                self._flush(st)
                st["n"] = -1  # done; ignore any further rows
        return logits

    def _flush(self, st):
        steps = sorted(st["recs"])
        recs = [st["recs"][t] for t in steps]
        out = {"step": np.array(steps, np.int32),
               "top_lp": np.stack([r["top_lp"] for r in recs]),
               "top_id": np.stack([r["top_id"] for r in recs]),
               "inp": np.array([r["inp"] for r in recs], np.int64),
               "sparse": np.array([r["sparse"] for r in recs], bool)}
        if st["forced"] is None:
            out["token"] = out["top_id"][:, 0].astype(np.int64)  # greedy reference path
        else:
            out["token"] = np.array(st["forced"][:len(steps)], np.int64)
            out["q_ref"] = np.stack([r["q_ref"] for r in recs])
        os.makedirs(os.path.dirname(st["out"]), exist_ok=True)
        tmp = st["out"] + ".tmp.npz"
        np.savez(tmp, **out)
        os.replace(tmp, st["out"])
