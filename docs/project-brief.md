# AgentRelay Project Brief

## Decision

Develop AgentRelay as a systems paper project about **risk-bounded transfer of obligation-closed semantic continuations**, not as another step-level LLM router or generic agent transaction runtime.

## Target And Mode

- Working label: **AgentRelay v2: Evidence-Carrying Semantic Continuations for Risk-Bounded Edge-Cloud Handoffs**
- Primary lens: ML systems / edge-cloud serving
- Provisional target: MLSys-style systems venue; TSC is a possible extended fallback
- CCFA mode: standard
- Development label: salvageable and novelty-grounded, with empirical go/no-go gates

## Raw-Idea Diagnosis

The supplied seed proposed step-criticality-aware device-cloud relay. Current literature invalidates a first-step-routing claim: Hera and AgentRouter already route long-horizon agent trajectories at step granularity, SWE-Router uses partial trajectories before escalation, and recent harness-native routing conditions on full execution state.

The surviving research ingredient is the **handoff itself**. Existing routers commonly abstract switching as a model-selection event. In deployment, a switch also requires state serialization, transport, target-model rehydration, semantic continuity checks, and safe handling of already executed tool effects. These costs and correctness obligations can reverse the apparent benefit of a model switch.

Adjacent work narrows the claim further. Arca already provides portable process continuations; Portable Agent Memory already addresses provenance-aware heterogeneous memory transfer; Cordon and Atomix already provide transaction/effect abstractions for agents; and Causal Past Logic already provides causal runtime verification for distributed LLM-agent workflows. AgentRelay therefore cannot claim any of these mechanisms in isolation. Its defensible object is their unresolved coupling during a heterogeneous model switch.

## Optimized Idea Card

- Task: execute a long-horizon tool-using agent across a small edge model and a stronger cloud-equivalent model.
- Audience: researchers building model routers, compound AI systems, agent runtimes, and edge-cloud LLM services.
- Gap hypothesis: model routing, semantic-state transfer, causal verification, and effect progress are studied largely as separate layers; the retained literature does not jointly compute the smallest valid state transfer and use its measured cost/risk to decide a heterogeneous model switch.
- Root challenge: heterogeneous models cannot share hidden state; compact payloads may omit a causal predecessor required by the next action; and switching during an indeterminate external effect can duplicate or invalidate a mutation.
- Core insight: the transferred object is an **obligation-closed semantic continuation** of a partially replicated state machine, not merely a summary or generic transaction record.
- Proposed mechanism: jointly choose executor, continuation mode, commit mode, and dwell decision; transfer a dependency-closed evidence-carrying packet; validate and selectively patch named predecessors; and restrict migration using an effect frontier.
- Strongest contribution types: new systems problem/formulation plus an integrated runtime mechanism.
- Expected evidence: ContinuationTax phase diagrams, routing-reversal analysis, fidelity/repair microbenchmarks, calibrated-risk diagnostics, effect-frontier failure injection, and Pareto evaluation on public benchmarks.
- Why now: 2026 work makes per-step agent routing practical, exposing state-continuity and effect-safety costs that coarse routers could previously ignore.
- Main risk: reviewers may view the system as Hera + Arca + RAMP + Cordon/Atomix.
- Risk-reduction condition: demonstrate non-additive interactions, especially decisions that change only after measured continuation size, patch probability, calibrated risk, and effect-frontier legality are considered, and show that no isolated component matches the joint system.

## Four Planned Innovations

### C1. ContinuationTax Phase Diagram

Measure encode, transferred bytes/tokens, recorded-network time, target prefill/rehydration, closure verification, selective patch, and effect-frontier wait/reconciliation. Map the phase boundary where switching becomes harmful as continuation size, model gap, network conditions, patch probability, and effect status vary. The claim remains conditional until measurements cause validated route reversals or expose a stable phase boundary.

Innovation type: systems measurement and problem formulation.

### C2. Obligation-Closed Evidence-Carrying Continuation

Represent goal constraints, evidence/provenance, world/tool state, plan obligations, trace references, and effect records as a dependency graph. For declared next-step obligations, compute a minimal supported dependency-closed cut. The packet carries hashes, dependency identifiers, schema/world versions, and effect frontier. The target validates closure and requests named missing predecessors instead of replaying the full trace.

Innovation type: model-independent semantic-state abstraction and transfer protocol.

### C3. Calibrated Risk-Bounded Semi-Markov Joint Router

At each step, select `(executor, continuation_mode, commit_mode, dwell_decision)`. Minimize measured latency, transfer, and cloud use subject to calibrated state-failure, effect-failure, and task-success constraints. Include switch amortization, hysteresis/minimum dwell, and conservative abstention to staying or full replay. Fit a lightweight controller on official training rollouts and calibrate on validation only.

Innovation type: constrained joint control over compute and semantic-state transfer.

### C4. Effect-Frontier-Coupled Migration Legality

Track effect progress as intent/prepared, sent, acknowledged, committed, compensated, or indeterminate. Make that frontier constrain whether migration is legal and which continuation/commit modes are available. Reconcile lost responses before retry and never regenerate a committed effect. The novelty claim is this coupling to continuation transfer and routing; the transaction machinery itself is prior art.

Innovation type: reliability constraint integrated into heterogeneous handoff control.

## Falsifiable Hypotheses

- H1: measured ContinuationTax causes a non-trivial set of executor decisions to differ from a model-only or zero-transfer-cost router.
- H2: an obligation-closed continuation preserves more downstream success per transferred token/byte than size-matched narrative summaries, edge-free typed deltas, and full-history truncation.
- H3: a frozen calibrated joint router improves at least one success-latency-transfer Pareto region while meeting predeclared validation risk targets.
- H4: coupling migration legality to the effect frontier reduces duplicate/conflicting mutations under controlled retry/handoff failures relative to a routing-independent barrier.

These are hypotheses, not results.

## Go/No-Go Gates

1. **Continuation-tax gate:** switch cost must either change at least 10% of candidate decisions or expose a repeatable phase boundary in at least two benchmark/model-network settings. Otherwise narrow the project to verified state continuity rather than switch-aware routing.
2. **Closure/fidelity gate:** the obligation-closed packet plus patch must outperform a size-matched narrative summary or typed delta without dependency closure in invariant recovery and downstream continuation on public validation data. Otherwise stop the four-contribution framing.
3. **Reliability gate:** the benchmark adapter must expose deterministic mutation identifiers or state diffs. If not, C4 is evaluated only where official sandbox state supports it and stated as scoped.
4. **Compute gate:** a complete pilot pair must run within 24 GB on the 4090D with reproducible configurations. If 14B plus edge model is unstable, execute models sequentially and report measured cold/warm handoff separately.

## Non-Goals

- No claim of first step-level agent routing.
- No claim of first portable continuation, portable agent state, causal agent verifier, or transactional agent runtime.
- No “proof-carrying” terminology without a defined proof object, checker, and soundness statement.
- No proprietary cloud API is required for the main result.
- No new synthetic task dataset.
- No full-parameter LLM training.
- No claim of universal transaction semantics for arbitrary real-world services.
- No fabricated or manually edited result values.

## Experimental Integrity Boundary

Only official public benchmark tasks and official splits are used. Native inference over the official training split may produce execution traces and routing labels; those traces are recorded run artifacts and never replace or alter the underlying tasks. All formal main-paper numbers originate from AutoDL RTX 4090D runs and immutable result files.

