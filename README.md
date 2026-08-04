# AgentRelay

CCFA-managed research project for a step-criticality-aware edge-cloud handoff runtime for agent execution.

## Current status

- Stage: implementation and software verification
- Active gate: AutoDL RTX 4090D environment installation and official-dev rollout
- Provisional venue: MLSys 2027, with TSC as a possible extension/fallback
- Local verification: RTX 4080 Laptop, small-batch debugging only
- Formal experiments: AutoDL RTX 4090D (24 GB)
- Dataset policy: real public datasets and official splits only
- Storage on AutoDL: `/root/autodl-tmp/AgentRelay`

No paper result is claimed from the local run. Training/calibration traces may
only use declared public train/development splits. The local diagnostic reads
one public AgentProcessBench `bfcl/test` input prefix after dropping all label
columns; it is never used for tuning, model selection, or manuscript tables.

## Verified local gate

- GPU: NVIDIA GeForce RTX 4080 Laptop GPU
- Native model: `Qwen/Qwen2.5-1.5B-Instruct@989aa798...aa306`
- Public data: `LulaCola/AgentProcessBench/bfcl@test@cd81f326...a98bf`
- Result: 1/1 nonempty native generation; semantic protocol v2, obligation
  closure, named predecessor patch, effect frontier, and trace provenance valid
- Software: 35 tests pass; 12 routing methods and 7 continuation codecs implemented
- Independent audit: all 16 checks passed; `paper_evidence=false`
- Peak CUDA memory: 9,908,711,424 bytes; generation latency: 8,791.74 ms
- Run: `artifacts/local-data/runs/local-smoke-20260804T133757296165Z-1a137502`

The next action requires an AutoDL RTX 4090D instance. See
`docs/autodl-handoff.md`; no formal benchmark has been started yet.

## Layout

```text
configs/       Reproducible experiment configurations
docs/          Project brief, method specification, and design decisions
experiments/   Experiment protocol, run manifests, and generated summaries
figures/       Manuscript figures generated from recorded data
literature/    Search log, screened papers, and novelty map
manuscript/    LaTeX manuscript and bibliography
reviews/       Review reports and revision ledger
scripts/       Entry points for profiling, training, and evaluation
src/           AgentRelay implementation
submission/    Venue checks and release package notes
tables/        Manuscript tables generated from recorded data
tests/         Unit and integration tests
```

Project state is tracked in `ccfa.yaml`.

The executable core includes obligation-closed content-addressed continuations,
named-predecessor patching, calibrated risk bounds, semi-Markov executor/state/
commit routing, effect-frontier migration masks, native Hugging Face execution,
official ALFWorld/WebShop/AppWorld adapters, immutable run manifests, runnable
baselines, and paired statistical analysis. See `docs/execution-guide.md` for
the strict local/formal separation.
