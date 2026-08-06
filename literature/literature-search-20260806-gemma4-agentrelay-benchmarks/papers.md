# Literature Search: Gemma 4-Compatible Benchmarks and Cross-Domain Routing Mechanisms

Date: 2026-08-06  
Purpose: select public benchmarks that place Gemma 4 E4B and 12B in a learnable edge/cloud regime, and identify non-neighboring mechanisms that strengthen AgentRelay without broad routing claims  
Target lens: MLSys / systems-oriented ML  
Source policy: primary scholarly or official project sources; current benchmark/model facts checked against official pages; MDPI and untraceable secondary sources excluded

## Executive Decision

- **Primary capability gate: tau2-bench text mode.** The official Gemma 4 model card reports 42.2% for E4B and 69.0% for 12B averaged over three Tau2 domains. This is the only retained agentic benchmark with direct, same-family evidence that both models are away from the 0% floor and that the gap is large enough to support cloud rescue.
- **Primary local-only fallback and second formal workload: InterCode-SQL.** It has deterministic Docker execution, official data inherited from Spider, multi-step repair feedback, and no evaluator-side LLM. Its fit is an inference rather than a measured fact: Gemma 4 scores 52.0%/72.0% on LiveCodeBench v6 for E4B/12B, so a paired InterCode pilot remains mandatory.
- **Preflight only: BFCL V4 non-live plus multi-turn.** It is useful for detecting tool-schema, chat-template, thinking-trace, and parser failures before an expensive rollout. It is not selected for supervised router training because the public release is primarily an evaluation benchmark and Gemma 4 pair results are not yet available.
- **Controlled mechanism calibration only: TextWorldExpress.** It offers explicit difficulty control and clean train/dev/test object splits, but its procedural nature conflicts with using it as the sole realism claim. Under the current project rules it is auxiliary, not primary paper evidence.
- **Retain WebShop only as a hard OOD/stress workload.** The audited 6%/8% gate is below the declared learnability band despite useful reward disagreement; it should not remain the router-training anchor.
- **Defer ToolSandbox, CAR-bench, WebArena, AppWorld, and SWE-bench.** They add simulator/judge cost, integration burden, or a likely small-model floor before the main method has passed a moderate-difficulty gate.

## Why Tau2 Changes the Dataset Decision

The official Gemma 4 evidence is unusually well aligned with this project:

| Evidence | E4B | 12B | Interpretation |
| --- | ---: | ---: | --- |
| Tau2, average over three domains | 42.2% | 69.0% | Desired middle band: edge is useful, cloud has meaningful rescue headroom |
| LiveCodeBench v6 | 52.0% | 72.0% | Supports an InterCode pilot, but does not directly establish interactive-code performance |
| GPQA Diamond | 58.6% | 78.8% | Confirms a material reasoning gap; not itself an agent benchmark |

Tau2 also exposes the state required by AgentRelay: policy text, conversation history, tool schemas, tool results, mutable world state, and—within Telecom—dual control by the user and agent. The current repository exposes a Gymnasium-compatible step interface and named task splits.

The main operational caveat is material: the published Tau2 setup uses an LLM user simulator (the paper reports GPT-4.1). A formal paired run must freeze the user-simulator model, version, prompt, seed, and decoding settings across both endpoints. If that API dependency is unavailable, InterCode-SQL becomes the first formal gate rather than silently substituting a weaker user simulator.

## Dataset Candidate Matrix

Scores are 1 (weak) to 5 (strong) and are screening judgments, not experimental results.

| Dataset | Direct pair-fit evidence | Sequential state | Deterministic / executable reward | Official split utility | 4090D practicality | Assigned role | Blocking issue |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| Tau2 text (retail/airline/telecom) | 5 | 5 | 5 | 4 | 3 | Primary capability gate and main agent workload | Fixed user simulator adds API/runtime cost |
| InterCode-SQL | 3 | 4 | 5 | 5 | 5 | Local-only fallback and second formal workload | Gemma pair fit must be measured, not inferred from LiveCodeBench |
| ScienceWorld official splits | 2 | 5 | 4 | 5 | 4 | Reserve long-horizon workload | Small-model success may still be near floor; integration is heavier |
| TextWorldExpress | 4 | 5 | 5 | 5 | 5 | Difficulty calibration and mechanism debugging | Procedural/toy environment; auxiliary evidence only |
| BFCL V4 non-live + multi-turn | 2 | 3 | 5 | 2 | 4 | Tool-call and parser preflight | No direct Gemma 4 result; evaluation-set training risk |
| WebShop | 1 | 4 | 5 | 5 | 3 | Hard OOD/stress workload | Observed 6%/8% success is below learnability band |
| AgentBoard wrapper | 2 | 5 | 4 | 2 | 3 | Analytics source, not router-training source | Public wrapper data are test-oriented; use underlying official splits instead |
| ToolSandbox | 1 | 5 | 4 | 3 | 2 | Deferred reliability stress | Complex tasks challenge frontier models; user simulation overhead |
| CAR-bench | 1 | 5 | 4 | 5 | 2 | Deferred ambiguity/limit-awareness stress | Best frontier Pass^3 is only 58%; likely E4B/12B floor and some judge dependence |
| TwinRouterBench static | 1 | 5 | 5 | 4 | 5 | External router diagnostic / prior-art baseline | Labels are tied to a locked 11-model tier pool, not the Gemma pair |

## Retained Papers and Official Sources

| # | Title | Year | Venue/source | Type | Insight | Completeness | Numeric evidence | Label | Relevance |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | [Gemma 4 Technical Report](https://arxiv.org/abs/2607.02770) and [official model card](https://ai.google.dev/gemma/docs/core/model_card_4) | 2026 | Google DeepMind / arXiv | system report | 5 | 5 | 5 | A | Direct E4B/12B capability evidence; Tau2 42.2/69.0 and LiveCodeBench 52.0/72.0. |
| 2 | [Tau2-Bench: Evaluating Conversational Agents in a Dual-Control Environment](https://arxiv.org/abs/2506.07982) | 2025 | arXiv + official code | benchmark/dataset | 5 | 4 | 4 | A | Dual-control tool use, executable final-state checks, compositional tasks, and controllable complexity. |
| 3 | [InterCode: Standardizing and Benchmarking Interactive Coding with Execution Feedback](https://proceedings.neurips.cc/paper_files/paper/2023/hash/4b175d846fb008d540d233c188379ff9-Abstract-Datasets_and_Benchmarks.html) | 2023 | NeurIPS Datasets & Benchmarks | benchmark/dataset | 4 | 5 | 4 | A | Deterministic interactive SQL/Bash/Python environments with execution feedback. |
| 4 | [The Berkeley Function Calling Leaderboard: From Tool Use to Agentic Evaluation](https://openreview.net/pdf/d52a12bb32128210600246f8979d90b892505cca.pdf) | 2025 | ICML | benchmark/dataset | 5 | 5 | 5 | A | Strong tool-call format, multi-turn, hallucination, and agentic preflight. |
| 5 | [AgentBoard: An Analytical Evaluation Board of Multi-turn LLM Agents](https://proceedings.neurips.cc/paper_files/paper/2024/hash/877b40688e330a0e2a3fc24084208dfa-Abstract-Datasets_and_Benchmarks_Track.html) | 2024 | NeurIPS Datasets & Benchmarks Oral | benchmark/dataset | 4 | 5 | 5 | A | Progress-rate and easy/hard analytics across nine agent tasks; split use must follow underlying environments. |
| 6 | [TextWorldExpress: Simulating Text Games at One Million Steps Per Second](https://aclanthology.org/2023.eacl-demo.20/) | 2023 | EACL Demo | system/tool | 4 | 4 | 4 | A | Fast simulator, train/dev/test object splits, and explicit room/distractor difficulty control. |
| 7 | [ToolSandbox](https://aclanthology.org/2025.naacl-findings.65/) | 2025 | NAACL Findings | benchmark/dataset | 4 | 5 | 4 | B | Stateful tool dependencies and dynamic milestones; likely too hard for the first pair gate. |
| 8 | [CAR-bench](https://arxiv.org/abs/2601.22027) | 2026 | arXiv preprint | benchmark/dataset | 4 | 4 | 4 | B | Public train/test tasks for ambiguity and limit-awareness, but current frontier difficulty is high. |
| 9 | [TwinRouterBench](https://arxiv.org/abs/2605.18859) | 2026 | arXiv preprint / workshop | method + benchmark | 5 | 4 | 5 | Risk | Execution-verified step downgrade labels and dynamic routing already exist; do not claim this labeling idea as new. |
| 10 | [LLMRouterBench](https://aclanthology.org/2026.findings-acl.1881/) | 2026 | Findings of ACL | benchmark/dataset | 5 | 5 | 5 | Risk | Over 400K instances, 21 datasets, and 33 models; many complex routers match simple baselines and remain far from oracle. |
| 11 | [AgentRouter](https://openreview.net/pdf?id=nu3GPfkyJV) | 2026 | ICML | pure method | 4 | 4 | 4 | Risk | Directly covers trajectory-aware step-level heterogeneous model routing. |
| 12 | [Meta-Router](https://iclr.cc/virtual/2026/poster/10007190) | 2026 | ICLR | pure method | 4 | 4 | 4 | Risk | CATE-based causal routing is already covered at query level; CATE alone is not a novelty claim. |
| 13 | [Consistent Estimators for Learning to Defer to an Expert](https://proceedings.mlr.press/v119/mozannar20b.html) | 2020 | ICML | theory/proof + method | 5 | 5 | 4 | A | Supplies the selective-deferral formulation: act locally or defer under asymmetric costs. |
| 14 | [On Stochastic Contextual Bandits with Knapsacks in Small Budget Regime](https://proceedings.iclr.cc/paper_files/paper/2025/hash/0a476350c56c221dc97fc024f4796e87-Abstract-Conference.html) | 2025 | ICLR | theory/proof + method | 4 | 5 | 4 | A | Supplies episode/global budget shadow pricing rather than independent per-step thresholds. |
| 15 | [FrugalGPT](https://openreview.net/forum?id=cSimKw5p6R) | 2024 | TMLR | pure method | 4 | 5 | 4 | A | Canonical learned cascade and mandatory simple baseline. |

`Risk` marks a direct claim boundary. `A` is a retained anchor. `B` is useful but deferred from the first formal gate. Scores summarize source-reported contribution quality and evidence, not paper acceptance probability.

## Cross-Domain Mechanisms to Integrate

### 1. Terminal-Reward Selective Deferral

Train the router on the downstream advantage

`Delta(s) = E[R_terminal(12B) - R_terminal(E4B) | state s]`

rather than model identity, next-action agreement, or absolute 12B success. Add continuation tax and tool-effect risk before escalating. Learning-to-defer is the mechanism anchor; Meta-Router means generic CATE language cannot be claimed as new.

### 2. Risk-Lower-Bound Escalation

Calibrate a lower confidence bound on net cloud advantage using train/calibration data only. Escalate only when the lower bound exceeds continuation and effect costs; otherwise stay on the current executor or abstain to a conservative full replay. The contribution must be evaluated as empirical risk control unless its assumptions and guarantee are explicitly proved.

### 3. Episode-Budgeted Semi-Markov Control

Replace an independent per-step cloud penalty with a dynamic shadow price for remaining cloud tokens/latency and remaining horizon. Add dwell time/hysteresis because every switch pays encode, transfer, rehydration, validation, and possible patch cost. Contextual bandits with knapsacks motivate the budget update; direct agent-routing work means the novelty must be the measured joint controller, not generic budget awareness.

### 4. Obligation-Closed Continuation and Effect-Frontier Legality

Retain the existing AgentRelay systems core: transmit only the dependency-closed state needed by declared next-step obligations, validate it at the target, selectively fetch missing predecessors, and disallow unsafe migration across unresolved irreversible effects. This is the differentiating systems object that direct router papers do not supply jointly.

## Predeclared Paired Gate

Run the same stratified task IDs, seeds, prompt, thinking mode, maximum steps, parser, and fixed user simulator for E4B and 12B. Select using train/dev only.

### Proceed as a cost-preserving deferral benchmark when all hold

- E4B terminal success/reward is in the 25%–65% band.
- 12B terminal success/reward is in the 50%–85% band.
- Cloud rescue rate, `P(12B succeeds and E4B fails)`, is at least 15%.
- E4B sufficiency, `P(E4B succeeds)`, is at least 25%.
- Paired reward disagreement is at least 25%.
- Invalid-action or harness-format failure is below 10% for each model.

### Additional requirement for a quality-improving routing claim

- The paired oracle exceeds always-12B by at least 3 percentage points on dev, or there is a separately powered continuous-reward oracle gap.

E4B-only wins are useful for quality improvement but are not required for cost-preserving deferral: a router can still be valuable if E4B solves many cheap cases and 12B rescues hard cases. This distinction prevents rejecting a valid edge/cloud cascade merely because the larger model dominates on average.

### Stop or change workload when any hold

- Both endpoints are below 15% success after harness errors are excluded.
- Both endpoints exceed 85% with less than 10% reward disagreement.
- More than 10% of episodes fail because of chat-template, tool-role, parser, or thinking-trace handling.
- A train/dev router cannot beat majority, task-type, and calibrated cascade baselines under paired bootstrap confidence intervals.

## Recommended Execution Order

1. BFCL V4: 20–30 non-live/multi-turn cases per model as a parser and native-function-call preflight; no router training.
2. Tau2: 50 paired tasks stratified across retail, airline, and telecom with a fixed official-style user simulator.
3. If Tau2 cannot be run with a fixed user simulator, run 50 paired InterCode-SQL tasks instead.
4. Only after a gate passes, collect train/calibration/dev trajectories and execution-verified one-step splice labels. Treat TwinRouterBench-style downgrade labeling as prior art and a training procedure, not an innovation.
5. Keep WebShop for final hard-OOD evaluation; do not spend the next cloud cycle fitting its near-floor training distribution.

## Claim Boundary

The broad routing field is crowded. Static/task-level routing, CATE routing, step-level agent routing, and execution-verified tier labels are already covered. The defensible project is therefore:

> risk-controlled, episode-budgeted sequential deferral coupled to obligation-closed semantic continuation and effect-safe migration between a fixed edge/cloud model pair.

This is an optimization target for implementation and experiments, not a frozen novelty claim. A full-text audit must be repeated before manuscript claims are written.
