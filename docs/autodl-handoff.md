# AutoDL RTX 4090D Handoff

This is the stop/go boundary requested by the user. The local RTX 4080 gate is
complete and independently audited. No paper draft work and no formal result
generation are authorized at this point.

## What passed locally

- 35 unit tests and Python source compilation.
- CUDA 12.8 tensor execution on an NVIDIA GeForce RTX 4080 Laptop GPU.
- One native Qwen2.5-1.5B generation over a public AgentProcessBench
  `bfcl/test` input prefix at immutable model/data commits.
- Semantic protocol v2 closure, selective predecessor patch, effect-frontier
  validation, and a measured-tax route-reversal fixture.
- 16/16 independent run checks, including source-tree match, packet/semantic
  hashes, trace provenance, environment metadata, output hash, label boundary,
  and non-evidence status.
- 12 routing methods and 7 state-transfer codecs are implemented; their results
  have not been generated locally.

Certified run directory:

```text
artifacts/local-data/runs/local-smoke-20260804T133757296165Z-1a137502
```

## Start on the 4090D instance

Upload `dist/AgentRelay-v2-autodl-source-20260804.tar.gz` and its `.sha256` sidecar
to the instance. Extract the archive directly onto the persistent data disk:

```bash
cd /root/autodl-tmp
sha256sum -c AgentRelay-v2-autodl-source-20260804.tar.gz.sha256
mkdir -p /root/autodl-tmp/AgentRelay
tar -xzf AgentRelay-v2-autodl-source-20260804.tar.gz \
  -C /root/autodl-tmp/AgentRelay
cd /root/autodl-tmp/AgentRelay
bash -n scripts/bootstrap_autodl.sh
```

Then run:

```bash
bash scripts/bootstrap_autodl.sh "$PWD"
source /root/autodl-tmp/AgentRelay/env.sh

python scripts/lock_config.py \
  configs/formal-autodl-4090d.template.json \
  /root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json

python scripts/preflight.py \
  /root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json \
  --output /root/autodl-tmp/AgentRelay/results/preflight.json
```

Send back the full contents of
`/root/autodl-tmp/AgentRelay/results/preflight.json`. The expected device name
must contain `4090`, CUDA must be available, GPU memory must be at least 20 GiB,
and every cache/temp path must remain under `/root/autodl-tmp/AgentRelay`.

After preflight, run the locked official checkout and environment preparation:

```bash
python scripts/download_public_data.py \
  /root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json
python scripts/checkout_repositories.py \
  /root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json
bash scripts/prepare_official_benchmarks.sh
```

Return the preflight JSON and the three checkout revision lines before starting
train/dev rollouts. Formal matrix commands and the train/dev/test boundary are
listed in `docs/execution-guide.md`.

## Deliberately not started yet

- Full ALFWorld, WebShop, or AppWorld runs.
- Formal baseline, ablation, robustness, or statistical matrices.
- Any result table or manuscript drafting.

The native adapters and baseline runner are present in the source package. The
remaining gate is empirical: install the pinned official environments, validate
one official train/dev task per adapter, profile the 4090D with a real recorded
network trace, then freeze the router and validation-only calibrator.
