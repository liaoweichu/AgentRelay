# AgentRelay Execution Guide

This guide separates software verification, local diagnostic inference, and
formal paper evidence. A command from one tier must never be relabeled as a
result from another tier.

## Tier 0: Software correctness (current machine)

```powershell
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
python -m compileall -q src scripts tests
python scripts/export_method_manifest.py \
  artifacts/local-data/results/implemented-methods-v2.json
python -m agentrelay.cli check-config configs/local-smoke.template.json --allow-unlocked
```

These tests use minimal program fixtures only. They test serialization,
transactions, routing arithmetic, and statistics; they are not benchmark data
and are never included in paper tables.

## Tier 1: RTX 4080 Laptop diagnostic probe

1. Install the `smoke` extra in a dedicated virtual environment, then replace
   any CPU PyTorch build with the official CUDA 12.8 wheel:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[smoke]"
.\.venv\Scripts\python.exe -m pip install --force-reinstall --no-deps `
  torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
```

2. Resolve mutable references into a locked local config:

```powershell
python scripts/lock_config.py `
  configs/local-smoke.template.json `
  configs/local-smoke.locked.json
python scripts/download_public_data.py configs/local-smoke.locked.json
python scripts/preflight.py configs/local-smoke.locked.json
python -m agentrelay.cli local-smoke `
  configs/local-smoke.locked.json --limit 1 --required-gpu-name 4080
python -m agentrelay.cli verify-run `
  artifacts/local-data/runs/<run-id> `
  --required-gpu-name 4080 --require-diagnostic-only --project-root .
```

When all assets are already cached, set `HF_HUB_OFFLINE=1`,
`TRANSFORMERS_OFFLINE=1`, and `HF_DATASETS_OFFLINE=1` to prevent metadata
network checks from obscuring an otherwise reproducible offline run.

Only the public AgentProcessBench `bfcl/test` input prefix and native
`Qwen2.5-1.5B-Instruct` inference are permitted in this tier. The adapter drops
`ground_truth`, `answer_text`, `step_labels`, `final_label`, and every other
non-allowlisted column before row iteration. Keep the one-row certified probe
and never exceed the configured eight-row debugging ceiling. Outputs must be
stamped `paper_evidence=false` and cannot populate manuscript result tables.

## Tier 2: AutoDL RTX 4090D formal runs

Copy the repository to the AutoDL instance, then run:

```bash
bash scripts/bootstrap_autodl.sh "$PWD"
source /root/autodl-tmp/AgentRelay/env.sh
python scripts/lock_config.py \
  configs/formal-autodl-4090d.template.json \
  /root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json
python scripts/preflight.py \
  /root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json \
  --output /root/autodl-tmp/AgentRelay/results/preflight.json
python scripts/download_public_data.py \
  /root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json
python scripts/checkout_repositories.py \
  /root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json
bash scripts/prepare_official_benchmarks.sh
```

The locked configuration records exact model, dataset, and official repository
commits. All datasets, models, native traces, checkpoints, logs, and results stay
under `/root/autodl-tmp/AgentRelay`.

The official adapters and common matrix runner are implemented. Before any
full evaluation, profile both pinned models on the 4090D and provide a real
recorded network trace; a null or default bandwidth is rejected:

```bash
python scripts/profile_models.py \
  /root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json \
  /root/autodl-tmp/AgentRelay/results/service-profile.json \
  --repeats 3 \
  --network-trace /root/autodl-tmp/AgentRelay/datasets/network/trace.csv \
  --rate-column mbps --sample-period-ms 1000 \
  --trace-source '<public trace citation and immutable revision>'
```

Create immutable complete-split manifests with
`scripts/build_official_task_manifest.py`. Use official train manifests first
to collect native fixed/cascade/full-replay rollouts, then build and fit the
router with `build_router_rows.py` and `fit_router.py`. Use only official
validation/dev manifests for `build_calibration_rows.py` and
`calibrate_router.py`. The task manifest hash, model/repository revisions,
profile hash, source-tree revision, and per-task result hashes are recorded by
`run_autodl_matrix.py`.

Example frozen evaluation call after those gates pass:

```bash
python scripts/run_autodl_matrix.py \
  /root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json \
  /root/autodl-tmp/AgentRelay/results/manifests/alfworld-valid-unseen.json \
  /root/autodl-tmp/AgentRelay/results/service-profile.json \
  --router /root/autodl-tmp/AgentRelay/results/router.joblib \
  --calibrator /root/autodl-tmp/AgentRelay/results/calibration.json \
  --method edge_only --method cloud_only --method agentrouter_style \
  --method hera_agreement --method full_replay_step \
  --method uncalibrated_joint --method agentrelay
```

The implemented method manifest contains 12 routing methods and 7 continuation
codecs. “Style” methods are local, paper-faithful operationalizations and must
not be described as exact official reproductions.

## Required ordering

Run the experiment gates in this order:

1. E0 software accounting and local diagnostic probe.
2. E2 state fidelity on the ALFWorld development split.
3. E4 effect-failure injection on the AppWorld development split.
4. Freeze schema, prompts, thresholds, and router configuration.
5. E1/E3 full official evaluation and routing-reversal analysis.
6. E5 real recorded network trace replay.
7. E6 cross-family generalization only if the declared budget remains.

Do not inspect test evaluator internals or alter prompts, selection thresholds,
or task order after a formal test run. Any interrupted run is retained with its
manifest and rerun from the beginning under a new run identifier.
