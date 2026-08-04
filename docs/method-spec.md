# AgentRelay Method Specification

Status: design specification; no result or guarantee is asserted.

## 1. Problem Formulation

An agent executes an official public benchmark task over steps `t = 1..T`. At step `t`, the runtime observes harness state `x_t`, previous executor `m_(t-1)`, the target's last acknowledged continuation version, dependency graph `G_t`, and effect frontier `F_t`. It chooses:

```text
a_t = (executor m_t,
       continuation mode z_t,
       commit mode g_t,
       dwell decision d_t)
```

where:

- `m_t` is an edge or cloud-equivalent open model;
- `z_t` is `reuse`, `closed_delta`, `closed_delta_patchable`, or `full_replay`;
- `g_t` is `immediate`, `barrier`, `reconcile`, or `compensating` when an effect is involved;
- `d_t` applies minimum dwell/hysteresis to avoid uneconomic switch churn.

The measured cost is:

```text
C_t = inference_latency
    + switch_indicator * (
        encode_latency
        + transfer_latency(bytes, recorded_bandwidth)
        + target_rehydration_latency
        + closure_verification_latency
        + expected_patch_latency
        + effect_frontier_wait_or_reconcile_latency
      )
    + token_or_cloud_use_cost
```

The controller minimizes cumulative measured cost subject to empirically calibrated task-success, continuation-fidelity, and effect-risk constraints. These are design constraints, not formal safety guarantees.

## 2. Semantic Continuation Model

### 2.1 Dependency graph

At each step, the runtime maintains `G_t = (V_t, E_t)`. Nodes are typed and content-addressed:

- `goal_constraint`: task goal, hard constraint, permission, success criterion;
- `evidence`: a value plus source span and provenance hash;
- `world_state`: observable environment/tool version, inventory, resource, or object state;
- `plan_obligation`: active subgoal, unresolved obligation, pending action, or completed decision;
- `trace_span`: immutable raw observation, reasoning-visible context, action, or tool result reference;
- `effect_record`: effect identity, status, arguments, result lineage, and recovery/compensation metadata.

An edge `u -> v` means node `v` semantically depends on `u` for target-side continuation or validation. Benchmark adapters declare observable dependencies and invariants; they may not inspect test labels or hidden evaluator outcomes.

### 2.2 Next-step obligations

`O_t` is the set of nodes the declared next operation must satisfy. Obligations are derived from public harness state, pending tool schemas, unresolved constraints, and effect progress. They are never derived from an official test answer or post-hoc success label.

### 2.3 Obligation-closed cut

A candidate continuation `S_t` is legal when:

```text
O_t subset_of S_t
required_predecessors(O_t, G_t) subset_of S_t or explicitly_patchable
world_version(S_t) == target_expected_version
migration_legal(F_t, g_t) == true
```

The packet builder seeks the smallest schema-valid `S_t` under the selected continuation mode. “Smallest” refers to the implemented dependency closure and byte/token objective, not an unproven globally minimal semantic representation.

## 3. Runtime Components

```text
Public benchmark environment
        |
        v
Harness collector ---> Dependency / obligation registry
        |                           |
        v                           v
Immutable trace store ---> Closure-cut packet builder
        |                           |
        v                           v
Effect interceptor ---> Effect frontier / migration guard
                                    |
                                    v
Measured-cost + calibrated-risk joint controller
                                    |
                                    v
                       target validate -> patch -> act
```

## 4. Evidence-Carrying Continuation Packet

Every packet is versioned, content-addressed, and serializable without a model-specific tokenizer.

```text
SemanticContinuationPacket
  schema_version
  task
    dataset_id
    dataset_revision
    split
    sample_id
    goal_hash
  source_and_target
    source_executor
    target_executor
    parent_packet_hash
    acknowledged_version
  obligations
    obligation_ids
    required_invariants
  nodes
    node_id
    node_type
    value_or_trace_ref
    provenance_hash
    world_version
  dependency_edges
  patchable_predecessor_ids
  effect_frontier
    effect_key
    status
    result_lineage_hash
    recovery_or_compensation_ref
  packet_hash
```

Full raw observations, model inputs/outputs, actions, and tool results remain in an immutable trace store. A delta contains only changed or newly required nodes plus dependency identifiers needed to validate closure.

“Evidence-carrying” means the packet contains operationally checkable hashes, versions, provenance, and dependencies. It is not described as proof-carrying unless a formal logic, proof object, checker, and soundness statement are added.

## 5. Validate-And-Selective-Patch Protocol

1. The source derives `O_t` and constructs a dependency-closed cut against the target's acknowledged version.
2. The target validates schema, content hashes, dependency closure, required invariants, world version, and effect frontier.
3. A benchmark adapter executes deterministic checks when the public environment exposes them; otherwise the check is explicitly labeled heuristic.
4. If a predecessor or provenance span is missing, the target returns named node IDs or `trace_span_id` values.
5. The source returns only the requested predecessors and their closure; the target re-validates.
6. If validation still fails, the maximum patch rounds are reached, or predicted patch cost exceeds the replay threshold, the runtime abstains to full replay or staying on the source executor.

The protocol records transferred bytes, target-tokenized tokens, encode/verify/patch latency, patch precision, patch rounds, closure status, and fallback reason.

The selective-repair structure is inspired by atomic-visibility systems such as RAMP; AgentRelay's research question is its adaptation to heterogeneous semantic continuations and joint routing, not a new general transaction protocol.

## 6. Calibrated Risk-Bounded Joint Controller

### 6.1 Features

- step index, remaining budget, current executor, and consecutive steps on that executor;
- task/step type exposed by the harness, never test labels;
- input length, candidate cut size, dependency depth, obligation count, and estimated patch probability;
- model uncertainty or calibrated confidence when operationally available;
- effect class/frontier and migration-legality mask;
- measured warm inference, encode, transfer, rehydration, validation, patch, and reconciliation profiles;
- prior handoff outcome, patch rate, and dwell length.

### 6.2 Valid actions

- Staying on the same executor normally uses `reuse`.
- Switching uses `closed_delta`, `closed_delta_patchable`, or `full_replay`.
- An effect in `sent` or `indeterminate` state can force `reconcile` and mask migration.
- Irreversible operations can force `barrier`.
- Low-confidence states can abstain to staying or `full_replay`.

### 6.3 Initial policy

The first formal implementation uses a lightweight logistic model, GBDT, or small MLP plus risk calibration. It does not require LLM-scale reinforcement learning.

```text
predicted_utility = predicted_success
                  - lambda_latency * measured_or_predicted_latency
                  - lambda_transfer * transfer_tokens_or_bytes
                  - lambda_cloud * cloud_use

choose highest-utility legal action only if:
  state_failure_risk <= epsilon_state
  effect_failure_risk <= epsilon_effect
  success_lower_bound >= tau
otherwise abstain
```

Training uses official training-split native rollouts only. Calibration and all thresholds use official validation tasks only. Test outcomes, hidden evaluator state, and answers never enter features, prompts, thresholds, labels, or action masks. Report empirical coverage and risk violations; do not assert distribution-free guarantees unless their assumptions are implemented and verified.

### 6.4 Semi-Markov switch control

Switch decisions include dwell state. Minimum dwell and hysteresis amortize continuation setup costs and prevent alternating executors on adjacent steps. The `no_hysteresis` ablation isolates this contribution.

## 7. Effect Frontier And Migration Legality

The tool interceptor canonicalizes calls:

```text
effect_key = hash(task_id, tool_name, canonical_arguments, environment_version)
```

Effect statuses are:

```text
intent -> prepared -> sent -> acknowledged -> committed
                    \-> indeterminate -> reconciled -> acknowledged/committed
prepared/acknowledged -> compensated   (only when supported)
```

Rules:

- `read_only` effects may execute immediately but their observed world version remains a dependency.
- `reversible` mutations require recovery or compensation metadata when the official sandbox supports it.
- `irreversible` mutations require a commit barrier.
- `sent` or `indeterminate` effects block migration until reconciliation establishes whether the effect occurred.
- A committed `effect_key` cannot be regenerated after handoff.
- Unknown tools default to a barrier and are counted in coverage metrics.

These semantics adapt prior agent-transaction ideas. AgentRelay's scoped claim is that frontier status changes continuation content and routing legality. It does not claim universal exactly-once semantics for arbitrary external APIs.

## 8. Required Baselines

- edge-only and cloud-only;
- task-level RouteLLM-style routing;
- FrugalGPT-style cascade;
- model-only step classifier;
- Hera-style and AgentRouter-style routers when reproducible;
- oracle model-only router, clearly labeled non-deployable;
- full replay and truncated history;
- size-matched narrative summary;
- unverified structured snapshot;
- typed delta without dependency edges;
- dependency-closed packet without patch;
- fixed dependency-closed packet with patch;
- joint controller without risk calibration;
- routing-independent effect barrier.

## 9. Split, Provenance, And Compute Discipline

- Task records come only from official public datasets and official splits.
- Router-training traces come only from official training tasks and native inference.
- Validation tasks tune thresholds, risk calibration, and checkpoint selection.
- Test tasks are read only by the frozen pipeline; labels and evaluator state are inaccessible to prompts and routing features.
- Every trace row stores dataset/model revision, sample ID, split, seed, prompt hash, code version, command, hardware, and timestamp.
- On one 24 GB RTX 4090D, edge and cloud-equivalent models may be served sequentially. Warm service profiles, cold loads, and trace-driven network time are reported separately.

## 10. Failure Modes And Scoped Fallbacks

- **ContinuationTax is negligible:** remove switch-aware routing as a primary contribution and focus on verified state continuity.
- **Closure adds no benefit:** stop the four-contribution framing and retain deterministic snapshots only as engineering.
- **Obligations are environment-specific:** expose a small public adapter contract, report coverage, and scope claims to supported benchmark families.
- **Patch storms:** cap rounds, abstain to full replay, and report tail behavior.
- **Calibration misses risk targets:** use conservative fallback and report violations; do not hide failed constraints.
- **Effect identity is unavailable:** restrict C4 to AppWorld or other official sandboxes with observable state diffs.
- **Small model cannot consume structured state:** compare concise-text and JSON renderings with identical underlying nodes.
- **Single-GPU timing distorts deployment:** separate native per-model inference from public recorded-network replay and label both precisely.

