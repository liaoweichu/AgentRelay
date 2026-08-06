# AgentRelay Experiment Plan

Mode: design + result-template  
Venue lens: ML systems / edge-cloud agent runtime  
Status: planned; no result value has been generated

## 1. Claim-Evidence Matrix

| Claim | Reviewer question | Required evidence | Benchmarks | Strong baselines | Primary metrics | Status |
| --- | --- | --- | --- | --- | --- | --- |
| C1 ContinuationTax | Does complete continuation overhead change a routing decision or define a repeatable no-switch region? | Component latency/bytes, phase diagram, and paired zero-tax vs measured-tax native route validation | ALFWorld, WebShop, AppWorld | model-only step router, AgentRouter-style classifier, Hera-style router | route-reversal rate, phase boundary, tax share, p50/p95 latency, switch churn | planned |
| C2 Obligation-closed continuation | Does dependency closure preserve what the target needs more efficiently than summaries or typed fields alone? | Size-matched forced-handoff comparison, missing-predecessor diagnostics, and selective-patch analysis | all three; AgentProcessBench for local diagnostics only | full replay, truncation, summary, snapshot, typed delta without dependency edges, closed packet without patch | closure/invariant recall, provenance validity, downstream success, patch precision/bytes, transfer size | planned |
| C3 Calibrated joint routing | Does joint executor/continuation/commit/dwell control improve the end-to-end frontier while respecting validation risk targets? | Frozen-policy full-split comparison plus calibration coverage/risk analysis | all three | edge/cloud-only, RouteLLM, FrugalGPT, model-only step router, Hera-style, uncalibrated joint router | success/reward, latency, cloud-step ratio, transfer, coverage, risk violations, dwell, Pareto frontier | planned |
| C4 Effect-frontier coupling | Does making effect progress constrain migration prevent duplicate/conflicting mutations beyond a routing-independent barrier? | Controlled interruption at effect lifecycle points in official sandbox tools | AppWorld primary; read-only control on ALFWorld/WebShop | no ledger, dedupe-only, routing-independent barrier, Atomix/Cordon-inspired frontier without joint routing, full AgentRelay | duplicate/conflicting effects, final-state correctness, blocked migrations, recovery latency | planned |

## 2. Dataset And Split Protocol

All tasks are real, public benchmark data. No task is invented, paraphrased, or modified. Native model rollouts on training tasks are run artifacts, not a replacement dataset.

### 2.1 Main benchmarks

#### ALFWorld

- Public benchmark scale: 3,827 tasks over six household activity categories.
- Data access: official ALFWorld assets; Hugging Face mirror may be used only after upstream file hashes and license are recorded.
- Training: complete official training split.
- Validation/testing: official `valid_seen` and `valid_unseen` splits; report both and their aggregate.
- Formal evaluation: full 140 seen + 134 unseen tasks when using the standard split exposed by the pinned distribution.
- Protocol aligned with Hera: maximum prompt 2,048 tokens, response 512 tokens, 50 environment steps, and up to 50 visible prior steps for full-history baselines.

#### WebShop

- Public benchmark scale: 1.18M real-world products and 12,087 crowd-sourced instructions.
- Data access: official Princeton WebShop repository and downloaded corpus.
- Use the official train/development/test manifests from the pinned release. Do not create a new random split.
- Formal evaluation uses the complete official test manifest; any inability to run it must be disclosed before results are used.
- Protocol aligned with Hera: maximum prompt 4,096 tokens, response 1,024 tokens, 20 environment steps, and up to 15 visible prior steps.

#### AppWorld

- Public benchmark scale: 750 tasks; official splits are 105 train, 60 dev, 168 test-normal, and 417 test-challenge.
- Data access: official AppWorld package and repository.
- Train split: demonstrations, policy fitting, and trace generation only.
- Dev split: threshold selection, ablation debugging, and error analysis.
- Test-normal/test-challenge: frozen pipeline evaluation only. Do not inspect task-wise evaluator output or tune after seeing results.
- Protocol aligned with Hera: maximum prompt 4,096 tokens, response 1,024 tokens, 50 environment steps, and up to 5 visible prior steps.

### 2.2 Diagnostic-only public dataset

- AgentProcessBench: 1,000 public trajectories, 250 each from BFCL, GAIA-dev, HotpotQA, and tau2.
- Use: local RTX 4080 schema/invariant and process-estimator debugging.
- Exclusion: its process labels are not used to tune or score the final main-benchmark test runs.

### 2.3 Real network traces

- Robustness uses public, recorded cellular traces only, such as Mahimahi's bundled Verizon LTE traces or a pinned CellReplay/public Dragonfly trace release.
- No randomly generated bandwidth trace is allowed.
- Report live local-GPU wall-clock latency separately from trace-driven communication latency.

## 3. Model And Hardware Protocol

### 3.1 Main open-model pair

- Edge executor: `google/gemma-4-E4B-it` at an immutable revision.
- Cloud-equivalent executor: `google/gemma-4-12b-it` at an immutable revision, using the declared memory-safe loading precision.
- Both use the same official chat template policy, thinking mode, token budget, greedy decoding, and seed. The config validator rejects any other formal model identity or unpaired decoding control.
- Rationale: the same Gemma 4 family reduces prompt/tokenizer confounds while preserving a measured capability gap; fixed endpoints run sequentially on a 24 GB 4090D.

### 3.2 Cross-benchmark generalization

- Keep the same frozen Gemma 4 E4B/12B pair.
- Test transfer across benchmark domains and difficulty bands rather than changing model families.
- Run after the primary benchmark and only if the full main protocol fits the formal budget.

### 3.3 Generation controls

- Use official chat templates and pin model/tokenizer revisions.
- Default formal decoding: greedy (`do_sample=false`) for deterministic tool actions unless a benchmark's canonical agent requires sampling.
- If sampling is required, use three predeclared seeds and identical seeds across paired methods.
- Methods share system prompt, task prompt, tool schema, maximum response length, stopping rules, and environment-step cap.

### 3.4 Hardware stages

- RTX 4080 Laptop: dependency checks, unit tests, and at most a small stratified debug batch. Results are diagnostic only.
- AutoDL RTX 4090D: all router fitting, complete native inference, main comparisons, ablations, stress tests, and reported measurements.
- Store models, datasets, traces, checkpoints, logs, and results under `/root/autodl-tmp/AgentRelay`.

## 4. Baseline Matrix

| Baseline | Why included | Implementation | Fairness constraints | Can run? |
| --- | --- | --- | --- | --- |
| Edge-only | minimum compute anchor | project runtime | identical prompt/tool budget | yes |
| Cloud-only | performance/cost anchor | project runtime | identical prompt/tool budget | yes |
| Random-0.3/0.5 | Hera-aligned sanity baseline | project runtime | same random seeds | yes |
| RouteLLM-style task router | standard coarse router | official RouteLLM components where compatible | train split only; same model pair | yes |
| FrugalGPT-style cascade | standard escalation baseline | faithful local reimplementation | same confidence data and token caps | yes |
| Model-only step router | isolates the original idea | same policy learner, executor-only action | no state/effect features | yes |
| AgentRouter-style classifier | closest multi-tier per-step baseline | implemented paper-faithful local operationalization | label as style baseline, not exact reproduction | yes |
| Hera-style router | closest device-cloud baseline | implemented action-agreement operationalization | same public tasks/model pair; not claimed as exact reproduction | yes |
| Full replay | transfer correctness anchor | full bounded task trace | same target token cap | yes |
| Narrative summary | common compressed handoff | frozen prompt and source model | size-matched to typed payload where possible | yes |
| Unverified structured snapshot | tests whether JSON alone suffices | same state fields, no validator/patch | same bytes/token budget | yes |
| Typed delta without dependency edges | isolates whether typed fields alone suffice | same changed fields, no predecessor graph | same renderer and target cap | yes |
| Dependency-closed packet without patch | isolates closure from repair | same graph/checker, direct replay fallback | same closure algorithm | yes |
| Uncalibrated joint router | isolates risk calibration/abstention | same action space and features, unconstrained utility | same training traces and learner capacity | yes |
| Routing-independent effect barrier | reliability baseline | progress barrier that never changes route/payload choice | same failure schedule | yes |
| Atomix/Cordon-inspired frontier | strongest transaction baseline | effect progress/recovery without joint continuation router | same effect schema and failure schedule | yes |

Externally reported baseline numbers are never mixed into the main table unless the model pair, prompt, dataset split, and metric definitions match exactly.

## 5. Metrics

### 5.1 Task quality

- official task success or benchmark reward;
- AppWorld task-goal and scenario-goal completion;
- invalid action rate and environment-step count;
- per-category/seen-unseen breakdown where official metadata permits post-hoc reporting.

### 5.2 Efficiency

- end-to-end wall-clock trajectory latency;
- p50 and p95 step latency;
- cloud-equivalent step ratio;
- model switches per trajectory, consecutive run/dwell length, and adjacent-step switch churn;
- transferred bytes and target-tokenized input tokens;
- peak GPU memory and controller CPU memory;
- controller, encode, verify, patch, rehydration, and inference time components.

### 5.3 ContinuationTax

```text
ContinuationTaxShare = (encode + communication + rehydration + closure_verification
                        + patch + effect_wait_or_reconciliation)
                       / end_to_end_latency
```

Report warm and cold model-loading cases separately. Main routing uses warm service profiles; cold load is a deployment sensitivity, not silently included. Plot route choice over continuation size, model gap, recorded-network condition, patch probability, and effect frontier to identify empirical phase boundaries.

### 5.4 State fidelity

- required-invariant recall and exact match;
- dependency-closure recall and missing-predecessor detection precision/recall;
- provenance/hash validity;
- stale-world-version rejection rate;
- successful first-pass rehydration rate;
- patch request rate, patch rounds, and patch bytes;
- patch precision: requested predecessors actually required by the checker;
- continuation success after a forced handoff.

### 5.5 Calibration And Control

- calibration coverage and abstention/fallback rate;
- empirical state-risk and effect-risk violations against predeclared validation targets;
- success lower-bound coverage where implemented;
- route churn, mean dwell, and blocked/overridden migration decisions;
- controller latency and memory.

### 5.6 Effect correctness

- exactly-once committed-effect rate;
- duplicate committed effects per trajectory;
- conflicting mutation rate;
- benchmark collateral-damage or state-test failures;
- recovery success and recovery latency.

## 6. Main Experiments

### E0. Local Continuation-Tax And Schema Gate

- Device: RTX 4080 Laptop.
- Data: a fixed, recorded debug manifest of public ALFWorld training tasks plus AgentProcessBench examples.
- Purpose: validate packet/dependency accounting and determine whether continuation tax is measurable.
- Not reportable as a formal result.
- Pass: the profiling and replay pipeline produces deterministic component totals and triggers the project gate defined in `docs/project-brief.md`.

### E1. End-To-End Main Comparison

- Device: AutoDL RTX 4090D.
- Data: full official evaluation splits for ALFWorld, WebShop, and AppWorld.
- Compare all runnable routing baselines plus AgentRelay.
- Plot task success against wall-clock latency, transfer tokens, and cloud-step ratio.
- Central test: C3.

### E2. Obligation Closure, Handoff Fidelity, And Repair

- Force a handoff at predeclared trajectory positions sampled from the official evaluation manifest without using outcome labels.
- Compare full replay, truncated history, narrative summary, unverified structured snapshot, typed delta without dependency edges, dependency-closed packet without patch, and dependency-closed packet + selective patch.
- Stratify by packet size, trajectory position, model pair, obligation count, dependency depth, and effect status.
- Report closure recall, missing-predecessor detection, patch precision/bytes, and downstream forced-handoff continuation.
- Central test: C2.

### E3. Routing-Reversal Analysis

- Replay identical frozen states through: (a) zero switch cost, (b) inference-only cost, and (c) measured full handoff cost.
- Count decisions and trajectory segments whose selected executor/payload changes.
- Validate changed choices by native execution, not only the utility predictor.
- Sweep continuation size, model gap, recorded-network condition, patch probability, and effect frontier to construct an empirical phase diagram.
- Central test: C1; this is the key defense against the "known-component assembly" criticism.

### E4. Handoff Failure Injection

- AppWorld only for mutation claims.
- Interrupt at: before send; after send/before response; after response/before acknowledgement; after acknowledgement/before continuation commit; and during target retry.
- Compare no ledger, dedupe-only, routing-independent barrier, Atomix/Cordon-inspired frontier without joint routing, and full AgentRelay.
- Record migration masks/overrides and reconcile indeterminate effects before retries.
- Central test: C4.

### E5. Recorded-Network Robustness

- Replay official payloads through pinned real cellular traces.
- Report by trace quartile and burst/drop segment, preserving original ordering.
- Test whether the policy adapts payload choice without changing benchmark data.
- Supports C1/C3; not a substitute for native inference.

### E6. Cross-Benchmark Generalization

- Freeze packet schema and policy form.
- Keep the frozen Gemma 4 E4B/12B pair and refit only authorized router/calibration parameters on the target benchmark's official train/dev splits.
- Evaluate a second admitted benchmark first; expand only if its paired capability gate passes and budget permits.
- Supports C2/C3 scope and generalization.

### E7. Microbenchmarks And Scaling

- Graph build, closure cut, encode, verify, and patch latency vs trace length, obligation count, and dependency depth.
- Payload bytes and token count vs trajectory length.
- Controller latency and memory vs action count, plus dwell/hysteresis overhead.
- Patch fallback threshold sensitivity.
- Supports implementation credibility.

## 7. Ablations

| Variant | Change | Mechanism tested | Expected interpretation after real results |
| --- | --- | --- | --- |
| Full AgentRelay | none | complete interaction | TBD |
| w/o ContinuationTax | remove transfer/rehydration/verification/patch/effect wait from policy | necessity of C1-aware routing | TBD |
| executor-only | choose model but fix full replay and commit behavior | necessity of C3 joint action | TBD |
| fixed closed delta | use one dependency-closed mode for every switch | joint policy vs continuation mechanism | TBD |
| typed delta, no edges | transmit the same typed fields without dependency graph/closure | unique value of C2 closure | TBD |
| closure, no patch | fall back directly to full replay on a missing predecessor | selective repair benefit | TBD |
| no provenance refs | values without source hashes/trace references | evidence validation contribution | TBD |
| uncalibrated utility | remove risk bounds and abstention | C3 calibration contribution | TBD |
| no hysteresis | allow adjacent unconstrained model switches | semi-Markov dwell contribution | TBD |
| no effect frontier | route without effect-state legality masks | C4 coupling necessity | TBD |
| routing-independent barrier | enforce the same barrier without changing route/payload | coupling vs generic transaction safety | TBD |
| dedupe only | hash duplicate calls, no frontier/reconciliation | progress tracking necessity | TBD |

## 8. Robustness And Failure Analysis

- long traces near context limit;
- small and large state deltas;
- shallow and deep obligation-dependency closures;
- stale packet/world versions;
- missing dependency predecessor or provenance span;
- one, two, and capped three patch rounds;
- low-bandwidth and burst-loss segments from real traces;
- the same frozen Gemma 4 pair across benchmark-specific prompt/tool schemas;
- read-only, reversible, irreversible, and unknown tool classes;
- prepared, sent, acknowledged, committed, and indeterminate effect frontiers;
- small-model schema-format failures;
- high-risk states that should trigger calibrated abstention/full replay;
- model-pair capability gap sensitivity;
- cases where full replay is correctly selected as the safest option.

Every failure category reports denominator and selection rule; examples are chosen by predeclared representative/hard/failure rules rather than cherry-picking.

## 9. Statistical Protocol

- Three independent runs for stochastic routing/training, matching Hera's reporting style.
- Pair task outcomes across methods using identical task order and seeds.
- Report mean and standard deviation across runs for peer comparability, plus task-level 95% paired bootstrap confidence intervals.
- For paired binary success, use McNemar's test; correct multiple primary comparisons with Holm's method.
- For latency and transfer size, report median, p95, and paired bootstrap intervals; do not rely on normally distributed assumptions.
- Predeclare primary comparisons: AgentRelay vs Hera-style, AgentRouter-style, model-only step router, and full-replay step router.
- Predeclare validation risk targets, calibration method, and coverage reporting before test execution.
- No claim of significance until the recorded result and test output exist.

## 10. Execution Priority

| Priority | Experiment | Claim | Cost | Dependency | Placement | Stop condition |
| --- | --- | --- | --- | --- | --- | --- |
| P0 | E0 local gate | C1/C2 feasibility | low | core runtime | internal | deterministic graph/packet accounting fails or continuation tax is immeasurable |
| P0 | E2 on ALFWorld dev | C2 | medium | dependency graph/validator/model adapters | main | obligation closure cannot beat size-matched summary or edge-free typed delta on fidelity |
| P0 | E4 on AppWorld dev | C4 | medium | tool interceptor | main | official state evaluator cannot identify effects |
| P1 | E1 ALFWorld full | C1/C2/C3 | high | frozen and calibrated P0 design | main | memory/runtime exceeds 4090D after sequential-service fallback |
| P1 | E1 WebShop/AppWorld full | C1-C4 | high | benchmark adapters | main | protocol cannot remain identical across baselines |
| P1 | E3 routing reversal | C1 | medium | main run traces | main | no validated decision reversal occurs |
| P2 | E5 real-network traces | C1/C3 robustness | medium | payload logs | appendix/main figure | trace provenance cannot be pinned |
| P2 | E6 same-pair cross-benchmark transfer | generalization | high | primary benchmark complete | appendix | target benchmark capability gate fails or formal compute budget is exhausted |
| P2 | E7 scaling | systems evidence | low-medium | runtime complete | main/appendix | none |

## 11. No-Fabrication Status

No experimental result has been generated here. Every `TBD` cell in project tables must be populated from immutable run outputs produced by the declared protocol or from a verified public baseline under an exactly matching setting.
