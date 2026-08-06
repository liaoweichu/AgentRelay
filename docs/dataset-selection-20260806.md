# Dataset Selection Decision for Gemma 4 E4B / 12B

Date: 2026-08-06  
Status: proposed execution decision; no formal benchmark result yet

## Decision

Use the following workload stack:

1. **Tau2 text mode** as the first paired learnability gate and primary agent workload.
2. **InterCode-SQL** as the local-only fallback and second deterministic workload.
3. **BFCL V4 non-live/multi-turn** as a tool-call harness preflight only.
4. **WebShop** as a hard OOD/stress workload, not the router-training anchor.
5. **TextWorldExpress** only for controlled mechanism debugging, not as sole formal evidence.

The reason to move away from WebShop as the training anchor is empirical: its audited E4B/12B success rates are 6%/8%, below the intended learnability band. In contrast, the official Gemma 4 model card reports 42.2%/69.0% on Tau2 averaged over three domains.

## Important Execution Constraint

Tau2's official-style setup uses a fixed LLM user simulator. A valid pair comparison must keep the simulator model/version, seed, prompt, and decoding identical across E4B and 12B runs. If that dependency cannot be frozen, use InterCode-SQL first; do not replace the simulator mid-experiment.

## Gate Manifest

For every candidate, hold constant:

- exact task IDs and official split;
- environment and dataset commit;
- E4B/12B model revisions and quantization;
- thinking mode and token budget;
- chat template, tool schema, parser, and error recovery;
- maximum steps and timeout;
- seed and, where applicable, user-simulator configuration.

Proceed when:

- E4B is 25%–65%;
- 12B is 50%–85%;
- cloud rescues at least 15% of tasks;
- E4B succeeds on at least 25%;
- paired reward disagreement is at least 25%;
- harness/format failures are below 10% per model.

For a claim of improving quality beyond always-12B, additionally require a dev oracle gap of at least 3 percentage points. Edge-exclusive wins are not required for a cost-preserving deferral claim.

## Method Consequence

Do not present task-level reward routing, CATE, step-level routing, or execution-verified downgrade labels as standalone innovations. Retain four coupled implementation targets:

1. terminal-reward selective deferral;
2. calibrated lower-bound escalation;
3. episode-budgeted semi-Markov control with dwell/hysteresis;
4. obligation-closed continuation plus effect-frontier migration legality.

Detailed evidence and claim boundaries are in `literature/literature-search-20260806-gemma4-agentrelay-benchmarks/papers.md`.
