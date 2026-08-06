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
python scripts/audit_model_pair.py
python scripts/preflight_tau2.py \
  configs/formal-autodl-4090d.locked.json \
  .tmp_tau2_src \
  configs/tau2-user-simulator.json \
  --software-only
```

These tests use minimal program fixtures only. They test serialization,
transactions, routing arithmetic, and statistics; they are not benchmark data
and are never included in paper tables.

## Tier 1: Local Model Probe Skipped

The local machine does not contain the Gemma 4 E4B/12B checkpoints and is not
authorized to download or run them. Local work stops after Tier 0 software and
model-policy checks. The first native Gemma smoke is G7 on the AutoDL 4090D,
using `docs/cloud-4090d-handoff-plan-20260806.md`.

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
bash scripts/prepare_tau2_bench.sh \
  /root/autodl-tmp/AgentRelay/repositories/tau2-bench
```

The locked configuration records exact model, dataset, and official repository
commits. All datasets, models, native traces, checkpoints, logs, and results stay
under `/root/autodl-tmp/AgentRelay`.

The only admissible formal identities are `google/gemma-4-E4B-it` for `edge`
and `google/gemma-4-12b-it` for `cloud`. The config validator, service-profile
check, run-context hash, endpoint collector, and router gate all enforce this
pair. A mutable `master` snapshot path is not an immutable model revision.

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

For the active Gemma 4 checkpoint, follow
`docs/cloud-4090d-handoff-plan-20260806.md`: build the deterministic τ²
train/dev manifest from official train, run the three-domain text smoke, run
the E4B precision sensitivity, then run the reward-aware router gate. Fixed
edge/cloud collections are separate processes so only one model is resident.

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

1. Software checks plus pinned τ²/ModelScope software preflight.
2. Three-domain τ² text smoke with the fixed user simulator.
3. E4B NF4/BF16-compute versus FP16 sensitivity diagnostic.
4. τ² official-train-derived train/dev reward-router learnability gate.
5. Freeze schema, prompts, thresholds, and router configuration.
6. Only after audit: formal evaluation, ablations, and cross-benchmark stress tests.

Do not inspect test evaluator internals or alter prompts, selection thresholds,
or task order after a formal test run. Any interrupted run is retained with its
manifest and rerun from the beginning under a new run identifier.
