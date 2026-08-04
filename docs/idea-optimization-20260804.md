# AgentRelay Idea Optimization: Adjacent-Mechanism Integration

Date: 2026-08-04  
Mode: CCFA standard idea optimization  
Status: research blueprint; not manuscript text and not an experimental result

## Decision

Refine AgentRelay into a system for **risk-bounded transfer of obligation-closed semantic continuations across heterogeneous LLM executors**.

The project should not compete on “another step router” or “another transaction runtime.” Its central question is:

> When an edge/cloud agent changes models mid-trajectory, what is the smallest causally and semantically sufficient continuation that the target can validate, repair, and execute without violating unresolved tool effects?

Working label: **AgentRelay v2: Evidence-Carrying Semantic Continuations for Risk-Bounded Edge-Cloud Handoffs**. This is a project label, not a frozen paper title.

## 1. Problem-Gap-Insight-Method Blueprint

### Problem

Long-horizon agents may switch between a small edge model and a stronger cloud-equivalent model at individual steps. Heterogeneous models cannot transfer hidden activations directly, so a switch requires an explicit continuation containing enough state for the target to act correctly.

### Verified prior-art boundary

- Step-level and trajectory-aware routing already exist.
- Portable/provenance-aware agent memory already exists.
- Agent transaction and effect-outbox runtimes already exist.
- Causal runtime verification for distributed agent workflows already exists.
- Portable operating-system continuations already exist.

### Gap hypothesis

Existing lines appear not to jointly solve four coupled decisions: which executor acts next, which minimal dependency-closed semantic state must cross the boundary, how the target verifies/repairs that state, and whether current effect progress permits migration/commit. This is an inference from the screened literature, not a proven universal absence claim.

### Root challenge

“Relevant context” is not equivalent to a short summary. A target needs a dependency-closed set of facts, constraints, provenance, world versions, pending obligations, and effect records. Sending too little causes semantic discontinuity; sending too much erases routing gains; switching during an indeterminate external effect can duplicate or invalidate a mutation.

### Core insight

Model handoff is best modeled as transfer of a **partially replicated semantic state-machine continuation**. The validity criterion is not narrative similarity but whether the received subgraph closes every declared next-step obligation and effect dependency.

### Method

1. Build a typed dependency graph from immutable trace spans and benchmark state.
2. Derive next-step obligations without using evaluation labels.
3. Compute an obligation-closed continuation cut.
4. Attach hashes, provenance, dependency identifiers, world version, and effect frontier.
5. Let the target validate closure and request named missing predecessors.
6. Jointly select executor, payload mode, and commit mode using measured costs and calibrated risk bounds.
7. Abstain to staying, full replay, or a commit barrier when confidence or legality checks fail.

## 2. Four Refined Innovations

### C1. ContinuationTax Phase Diagram

Measure the complete handoff cost:

```text
encode + bytes/network + rehydration/prefill + closure verification
+ selective patch + effect-frontier wait/reconciliation
```

Produce a phase diagram over continuation size, model capability gap, network trace, patch probability, and effect status. The falsifiable contribution is that real handoff cost changes the optimal route or makes switching harmful in identifiable regions.

Novelty type: systems measurement and problem formulation.  
Do not claim: first switch-cost measurement.

### C2. Obligation-Closed Evidence-Carrying Continuation

Represent live state as a dependency DAG/hypergraph whose nodes include hard constraints, evidence, world/tool state, plan obligations, raw trace spans, and effect records. For a declared next-step obligation set, compute the smallest supported dependency-closed cut under the chosen packet schema.

Each packet carries content hashes, provenance, dependency identifiers, schema/world versions, and effect frontier. The target checks structural closure and task-specific invariants. On a missing predecessor, it requests that predecessor rather than replaying the entire trace.

Novelty type: model-independent semantic-state abstraction plus transfer protocol.  
Do not claim: formal proof-carrying state unless a proof system and soundness result are implemented.

### C3. Calibrated Risk-Bounded Semi-Markov Joint Router

Choose:

```text
(next executor, continuation mode, commit mode, dwell decision)
```

Minimize measured latency/transfer/cloud use subject to calibrated lower bounds on continuation/task success and upper bounds on invariant/effect failure. Include switch amortization, minimum dwell or hysteresis, and fallback/abstention. Fit only a lightweight model on official training rollouts and calibrate on official validation tasks.

Novelty type: constrained joint control over compute and semantic state transfer.  
Do not claim: distribution-free safety unless the exact calibration assumptions and guarantees hold.

### C4. Effect-Frontier-Coupled Migration Legality

Track effect progress using established transaction ideas: intent/prepared, sent, acknowledged, committed, compensated, or indeterminate. Migration is allowed, delayed, or forced into a conservative payload/commit mode based on the frontier. A lost response must be reconciled before retry; committed effects are not regenerated.

The new claim is the coupling between effect progress, continuation content, and routing legality. The transaction machinery itself is prior art.

Novelty type: reliability constraint integrated into heterogeneous handoff control.  
Do not claim: first transactional agent runtime.

## 3. Why The Combination Is Potentially Non-Additive

The project survives the “known-component assembly” objection only if the data show interactions such as:

- a model-only router switches, but ContinuationTax makes staying optimal;
- a size-only compressor sends a short packet, but obligation closure predicts a patch or failure;
- an executor is otherwise optimal, but an indeterminate effect frontier makes migration illegal;
- risk calibration selects full replay only for hard/uncertain states, improving the frontier over a fixed codec;
- selective patch changes both the effective handoff cost and the next routing decision.

If these interactions are absent, the project should narrow to a state-continuity systems study rather than maintain four contribution claims.

## 4. Minimal Formal Objects

Let `G_t = (V_t, E_t)` be the dependency graph at step `t`, `O_t` the next-step obligation nodes, and `F_t` the effect frontier. A continuation cut `S_t` is legal only if:

```text
O_t subset_of S_t
dependencies(S_t, O_t) subset_of S_t or resolvable_by_patch
world_version(S_t) == target_expected_version
migration_legal(F_t, commit_mode) == true
```

The controller chooses `a_t = (m_t, z_t, g_t, d_t)`, where `d_t` is a dwell/stay decision, under empirical risk constraints:

```text
min E[latency + transfer + cloud_use]
s.t. estimated continuation-failure risk <= epsilon_state
     estimated effect-failure risk <= epsilon_effect
     estimated task-success >= tau
```

These are design objectives, not guarantees or results.

## 5. Evidence Plan

| Claim | Decisive evidence | Failure condition |
| --- | --- | --- |
| C1 | Measured route-reversal rate and phase boundary under native execution and public network traces | Handoff cost rarely changes a validated choice |
| C2 | Obligation-closed cut beats size-matched summary, snapshot, and typed delta without closure on fidelity per byte/token | Closure metadata adds overhead without continuation benefit |
| C3 | Frozen calibrated policy improves a success/latency/transfer frontier while meeting predeclared validation risk targets | Risk violations are uncontrolled or router cost dominates |
| C4 | Effect-frontier coupling prevents duplicate/conflicting mutations under official sandbox failure injection | Benchmark cannot expose deterministic effect identity/state diff |

Required diagnostics include closure recall, missing-predecessor detection, patch precision/bytes, calibration coverage, risk violations, switch churn, dwell length, route reversal, duplicate effects, and recovery latency.

## 6. Baseline Additions

- full replay;
- bounded/truncated history;
- narrative summary;
- structured snapshot;
- typed delta without dependency edges;
- dependency-closed packet without patch;
- dependency-closed packet with selective patch;
- executor-only router;
- joint router without calibrated constraints;
- joint router without dwell/hysteresis;
- effect barrier independent of routing;
- full effect-frontier-coupled AgentRelay.

Direct routing baselines remain Hera-style and AgentRouter-style implementations where reproducible.

## 7. 4090D Feasibility

- Serve the 1.5B edge and 7B/14B cloud-equivalent models sequentially on one 24 GB RTX 4090D if concurrent residency is unstable.
- Keep the controller lightweight: logistic/GBDT/small MLP plus calibration; no LLM-scale RL is required for the first formal result.
- Start with ALFWorld dev for continuation closure/fidelity and AppWorld dev for effect-frontier behavior; add WebShop after both gates pass.
- Use official train rollouts for fitting and official dev for calibration. Freeze all policies and thresholds before official test execution.
- AgentProcessBench remains diagnostic only and does not provide final router labels.
- No large-model fine-tuning is necessary. Optional LoRA is out of the critical path and must earn its compute through a separate gate.

## 8. Implementation Delta

The next implementation phase on AutoDL should add, in order:

1. dependency-node and obligation schemas;
2. closure-cut builder and named-predecessor patch protocol;
3. closure/fidelity instrumentation;
4. effect frontier and migration-legality checks;
5. measured cost profiler and hysteretic joint controller;
6. train/dev-only risk calibration and abstention;
7. ALFWorld/AppWorld adapters, followed by WebShop;
8. immutable result export and audit.

The existing local RTX 4080 certified smoke remains valid because this optimization document does not modify runtime source.

## 9. Reviewer Pre-Mortem

| Likely objection | Severity | Required response |
| --- | --- | --- |
| “This is Hera + Arca + RAMP + Cordon.” | critical | One formal continuation abstraction, joint objective, and non-additive interaction experiments |
| “Obligations are hand-authored benchmark features.” | high | Small public adapter contract; report adapter code/coverage and cross-benchmark common fields |
| “Calibration is tuned on test behavior.” | critical | Immutable split manifests, frozen thresholds, no evaluator/test-label access |
| “Evidence-carrying is proof-carrying branding.” | high | Use evidence-carrying terminology and make every checker operationally explicit |
| “Exactly-once is impossible for arbitrary APIs.” | high | Scope claims to official sandbox environments with observable identifiers/state diffs |
| “Single-GPU timing does not represent edge-cloud deployment.” | medium | Separate native inference, measured serialization, and public trace-driven communication; report warm/cold cases |

## 10. Go/No-Go Recommendation

Proceed to the 4090D implementation phase only with the refined C1-C4 statements. Before formal main runs, require three gates:

1. obligation closure improves forced-handoff continuation fidelity over a size-matched summary or edge-free typed delta;
2. measured continuation cost produces validated routing reversals or a clear phase boundary;
3. AppWorld exposes enough effect identity/state information for scoped failure-injection evaluation.

If gate 1 fails, stop the four-contribution framing. If gate 2 fails, remove switch-aware routing as a primary contribution. If gate 3 fails, keep effect-frontier logic as scoped engineering and do not make it a headline claim.

