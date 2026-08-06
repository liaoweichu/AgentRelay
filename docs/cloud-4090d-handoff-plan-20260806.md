# AutoDL 4090D Gemma 4 Execution Handoff

Date: 2026-08-06  
Execution host: AutoDL RTX 4090D, 24 GB  
Model pair: edge `google/gemma-4-E4B-it`; cloud `google/gemma-4-12b-it`  
Current objective: complete the cloud-only model gate and WebShop official
train/dev learnability gate. Do not run the official test split.

## Responsibility Boundary

- The local RTX 4080 performs software tests only. It does not download, lock,
  load, profile, or run either Gemma model.
- The 4090D performs model locking, native inference, service profiling, and
  train/dev rollouts.
- Do not reuse the old Qwen service profile or legacy matrix. Do not treat a
  ModelScope `snapshots/master` directory as an immutable model revision.
- Do not start until the model-policy changes are committed and pushed. The
  cloud operator must receive the exact commit as `<MODEL_POLICY_COMMIT>`.

## Execution Plan Table

| Gate | Cloud action | Required output | Pass condition | Stop / return condition |
| --- | --- | --- | --- | --- |
| G0 | Check out the exact handed-off Git commit | `git rev-parse HEAD` and clean `git status --short` | SHA equals `<MODEL_POLICY_COMMIT>`; working tree is clean | Any mismatch or cloud-local edit: stop; do not reset or overwrite it |
| G1 | Bootstrap the isolated runtime under `/root/autodl-tmp/AgentRelay` | activated venv and `env.sh` | installation completes; all cache/temp variables point below the project root | dependency or disk failure: return the complete error log |
| G2 | Run software and model-policy checks | unit-test log, compile status, model-pair audit JSON | 57 tests pass; compilation succeeds; audit says `valid=true` and exact E4B/12B IDs | any failure: stop before model loading |
| G3 | Resolve every mutable model/data/repository reference | `formal-autodl-4090d.locked.json` | both model revisions and every dataset/repository revision are full immutable hashes | HF/Git resolution or access failure: stop; never derive a fake hash from a local path |
| G4 | Run hardware/storage preflight | `results/preflight-gemma4.json` | GPU name contains `4090`; memory is at least 20 GiB; CUDA and bitsandbytes load; paths are confined to persistent storage | any failed assertion: stop |
| G5 | Download pinned data and check out official environments | dataset row/revision lines and three repository revision lines | every resolved revision equals the lock; WebShop preparation succeeds | missing or mismatched revision: stop |
| G6 | Sequentially smoke-test E4B and 12B through the tracked inference path | `results/service-profile-gemma4-smoke.json` | file contains exact IDs/revisions for both roles; both produce native nonempty output; only one model is resident at a time | OOM/load/parser failure: stop and return the file/log; do not change model, precision, thinking mode, or token budget |
| G7 | Generate a fresh Gemma service profile using the retained real network trace | `results/service-profile-gemma4.json` | exact IDs/revisions match the lock; non-null positive bandwidth; profile self-hash exists | old Qwen IDs, null bandwidth, or trace failure: stop |
| G8 | Run 200 paired official-train and 100 paired official-dev WebShop tasks | gate receipt, four endpoint runs, paired JSONL, router, gate JSON | command finishes with status 0 or declared gate-failure status 2; all provenance checks pass | status 2 means empirical gate failure: return artifacts and do not run test; any other error may be resumed only with the identical command |
| G9 | Return the audit packet | files listed below plus run-directory paths | hashes and paths are complete enough for local independent review | do not start E1-E7 or manuscript work before review |

## Step 0: Exact-Code Gate

Replace `<MODEL_POLICY_COMMIT>` with the commit supplied after the current local
changes are committed and pushed.

```bash
cd /root/autodl-tmp/AgentRelay
git status --short
git fetch origin
git checkout main
git pull --ff-only origin main
git rev-parse HEAD
git diff --exit-code
git diff --cached --exit-code
test "$(git rev-parse HEAD)" = "<MODEL_POLICY_COMMIT>"
```

If the first `git status --short` is nonempty, stop and report it. Do not use
`git reset --hard`, do not delete cloud files, and do not continue on a
different commit.

## Step 1: Bootstrap and Software Gate

```bash
cd /root/autodl-tmp/AgentRelay
bash scripts/bootstrap_autodl.sh "$PWD"
source /root/autodl-tmp/AgentRelay/env.sh

python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python scripts/audit_model_pair.py
```

Expected audit identities:

```text
edge  = google/gemma-4-E4B-it
cloud = google/gemma-4-12b-it
```

## Step 2: Create and Inspect the Immutable Lock

```bash
cd /root/autodl-tmp/AgentRelay
LOCK=/root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json

python scripts/lock_config.py \
  configs/formal-autodl-4090d.template.json \
  "$LOCK"

python -m agentrelay.cli check-config "$LOCK"
python -c 'import json,re; p=json.load(open("/root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json")); expected={"edge":"google/gemma-4-E4B-it","cloud":"google/gemma-4-12b-it"}; assert {k:v["model_id"] for k,v in p["models"].items()}==expected; assert all(re.fullmatch(r"[0-9a-fA-F]{40,64}",v["revision"]) for v in p["models"].values()); print({k:(v["model_id"],v["revision"]) for k,v in p["models"].items()})'
```

If Hugging Face access or license authorization prevents revision resolution,
stop and return the error. The existing ModelScope `master` snapshots may be
kept as caches, but they cannot replace the immutable lock.

## Step 3: Hardware and Storage Preflight

```bash
python scripts/preflight.py \
  "$LOCK" \
  --output /root/autodl-tmp/AgentRelay/results/preflight-gemma4.json
```

Return `preflight-gemma4.json` immediately if this step fails. Do not begin a
model download or rollout with a failed preflight.

## Step 4: Pinned Data and Official Environments

```bash
python scripts/download_public_data.py "$LOCK"
python scripts/checkout_repositories.py "$LOCK"
bash scripts/prepare_official_benchmarks.sh
```

Keep the complete stdout containing dataset row counts and repository commits.

## Step 5: Sequential Two-Model Smoke

This command loads E4B, releases it, then loads 12B. It is diagnostic only and
does not need a network trace.

```bash
python scripts/profile_models.py \
  "$LOCK" \
  /root/autodl-tmp/AgentRelay/results/service-profile-gemma4-smoke.json \
  --repeats 1
```

Validate identity against the lock:

```bash
python -c 'import json; root="/root/autodl-tmp/AgentRelay/"; c=json.load(open(root+"formal-autodl-4090d.locked.json")); p=json.load(open(root+"results/service-profile-gemma4-smoke.json")); assert all(p[r]["model_id"]==c["models"][r]["model_id"] and p[r]["model_revision"]==c["models"][r]["revision"] and p[r]["output_tokens_median"]>0 for r in ("edge","cloud")); print({r:(p[r]["model_id"],p[r]["model_revision"],p[r]["output_tokens_median"],p[r]["peak_cuda_memory_bytes"]) for r in ("edge","cloud")})'
```

Do not edit quantization or retry with another model after an OOM. Return the
error and peak-memory information so the formal config can be revised once.

## Step 6: Fresh Gemma Service Profile

The old service-profile JSON is forbidden as a latency/model profile; only its
`network_trace.source` provenance string may be read. The raw real network
trace may be reused only if
`/root/autodl-tmp/AgentRelay/datasets/network/trace.csv` still exists, its
`mbps` column is valid, and its original acquisition source is copied exactly.
The model file used to record network throughput is not an inference model and
does not alter the E4B/12B experiment pair.

```bash
test -s /root/autodl-tmp/AgentRelay/datasets/network/trace.csv
test -s /root/autodl-tmp/AgentRelay/results/service-profile.json
python -c 'import csv,statistics; p="/root/autodl-tmp/AgentRelay/datasets/network/trace.csv"; rows=list(csv.DictReader(open(p))); assert len(rows)>=30; values=[float(r["mbps"]) for r in rows]; assert min(values)>=0 and statistics.median(values)>0; print("trace_samples",len(values),"median_mbps",statistics.median(values),"max_mbps",max(values))'
TRACE_SOURCE="$(python -c 'import json; p=json.load(open("/root/autodl-tmp/AgentRelay/results/service-profile.json")); print(p["network_trace"]["source"])')"
test -n "$TRACE_SOURCE"

python scripts/profile_models.py \
  "$LOCK" \
  /root/autodl-tmp/AgentRelay/results/service-profile-gemma4.json \
  --repeats 3 \
  --network-trace /root/autodl-tmp/AgentRelay/datasets/network/trace.csv \
  --rate-column mbps \
  --sample-period-ms 1000 \
  --trace-source "$TRACE_SOURCE"
```

If the raw trace is absent or invalid, stop here and return that fact. Do not
insert a constant/default bandwidth and do not reuse the old Qwen profile.

## Step 7: WebShop Official Train/Dev Learnability Gate

Run this only after G0-G7 pass:

```bash
python scripts/run_webshop_train_dev_gate.py \
  --config "$LOCK" \
  --profile /root/autodl-tmp/AgentRelay/results/service-profile-gemma4.json \
  --webshop-file /root/autodl-tmp/AgentRelay/repositories/webshop/data/items_shuffle.json \
  --revision 64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd \
  --output-dir /root/autodl-tmp/AgentRelay/results/webshop-train-dev-gate
```

The four endpoint runs are sequential and single-model-resident. If SSH or the
process is interrupted, rerun exactly the same command and output directory;
the receipt and episode hashes control safe resume.

- Exit `0`: gate passed. Return artifacts; still do not run official test.
- Exit `2`: gate completed but failed empirically. Return artifacts and stop
  WebShop expansion; the next project decision is a Tau2/InterCode adapter gate.
- Any other exit: runtime/integrity failure. Return the log and receipt; do not
  reinterpret it as an empirical failure.

## Step 8: Required Return Packet

Return all of the following without editing their contents:

1. Exact Git commit from `git rev-parse HEAD`.
2. `formal-autodl-4090d.locked.json`.
3. `results/preflight-gemma4.json`.
4. `results/service-profile-gemma4-smoke.json`.
5. `results/service-profile-gemma4.json`.
6. `results/webshop-train-dev-gate/receipt.json`.
7. `results/webshop-train-dev-gate/router-webshop-train-dev-gate.json`.
8. Both paired endpoint JSONL files and the router metadata/artifact.
9. The four run directories named in the receipt, including every
   `run-context.json`, `manifest.json`, and episode result.
10. Complete console logs and the final process exit code.

Do not run official WebShop test tasks, the full baseline matrix, ablations, or
paper-result generation until this packet is independently reviewed.

## Proposed `ccfa.yaml` Update After Handoff

Do not apply this state change before the cloud results are returned and
audited. At that point the orchestrator should update:

```yaml
stage:
  gate: "webshop_train_dev_reward_router_gate_audited"
constraints:
  hardware:
    local_debug: "software_only_no_gemma_models"
artifacts:
  cloud_4090d_handoff: "docs/cloud-4090d-handoff-plan-20260806.md"
```

The next owner after the returned packet is `ccf-integrity-auditor`. If the
gate passes, hand off to `ccf-experiment-designer`; if it fails, hand off to
the implementation owner for the preselected Tau2/InterCode capability gate.
