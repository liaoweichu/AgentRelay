# Literature Search: Adjacent Mechanisms for AgentRelay

Date: 2026-08-04  
Purpose: use mechanisms from distributed systems, runtime verification, continuations, and risk-controlled decision making to refine AgentRelay  
Target lens: MLSys / systems-oriented ML  
Source policy: primary scholarly sources only in the retained set; low-quality, secondary, and untraceable sources excluded

## Executive Finding

- Screened candidates: 24; retained high-value papers: 15.
- The neighboring literature invalidates four broad novelty claims: first transactional agent runtime, first causal workflow verifier, first portable continuation, and first provenance-aware agent-memory transfer.
- The defensible research object is narrower and more integrated: a model switch transfers a **partially replicated semantic continuation** between heterogeneous executors.
- The proposed continuation must be dependency-closed for the next step, carry checkable evidence and effect progress, support selective repair, and be selected jointly with the executor and commit mode.
- Novelty remains a hypothesis until full-text prior-art review and component-isolating experiments are complete.

## Retained Papers

| # | Paper | Year/source | Role | What it establishes | Implication for AgentRelay | Label |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | [Hera: Learning Long-Horizon Coordination for Device-Cloud Collaborative LLM Agents](https://arxiv.org/abs/2605.24598) | 2026, arXiv | direct routing | Step-level device/cloud routing over long-horizon agent trajectories. | Do not claim first step-level edge/cloud router; model real handoff state and cost instead. | Risk |
| 2 | [AgentRouter: Heterogeneous Model Routing for Cost-Optimal Multi-Step Agentic Workflows](https://openreview.net/pdf?id=nu3GPfkyJV) | 2026, ICML/PMLR paper PDF | direct routing | Trajectory-aware per-step routing across heterogeneous model tiers. | A classifier-only or switch-frequency contribution is insufficient. | Risk |
| 3 | [Speculative Actions: A Lossless Framework for Faster AI Agents](https://openreview.net/forum?id=P0GOk5wslg) | 2026, ICLR Oral | execution | Speculative next actions can be validated before effectful commit. | Validation and effect classes are prior mechanisms; focus on migration coupling. | Risk |
| 4 | [Portable Agent Memory](https://arxiv.org/abs/2605.11032) | 2026, arXiv | state portability | Structured memory, provenance, integrity, and heterogeneous rehydration. | Do not claim first portable or cryptographically verified agent state. | Risk |
| 5 | [Atomix: Progress-Aware Transactional Runtime for Agent Tool Calls](https://arxiv.org/abs/2602.14849) | 2026, arXiv | effect transactions | Progress-aware transaction semantics for buffered, reversible, and irreversible effects. | Use effect progress as a routing legality constraint, not a standalone ledger novelty. | Risk |
| 6 | [Cordon: Semantic Transactions for Tool-Using LLM Agents](https://arxiv.org/abs/2606.17573) | 2026, arXiv | semantic transactions | Task-scoped transactions, result lineage, shadow state, effect outbox, validation, and recovery metadata. | Remove any broad “first transactional agent runtime” claim. | Risk |
| 7 | [Causal Past Logic for Runtime Verification of Distributed LLM Agent Workflows](https://arxiv.org/abs/2605.20923) | 2026, arXiv | runtime verification | Causal-past guards and vector-clock online monitoring for distributed agent workflows. | Causal verification alone is not novel; use causal closure to minimize transfer. | Risk |
| 8 | [SAGA: Workflow-Atomic Scheduling for AI Agent Inference on GPU Clusters](https://arxiv.org/abs/2605.00528) | 2026, arXiv | workflow scheduling | Treats the workflow as a schedulable unit and exploits execution-graph/session locality. | Avoid broad workflow-graph or workflow-aware scheduling claims. | Risk |
| 9 | [Continuation-Centric Computing with Arca](https://www.usenix.org/conference/osdi26/presentation/srivatsan) | OSDI 2026 | continuations | Serializable portable continuations can be paused, migrated, and copied. | Distinguish process continuation from model-semantic continuation. | A |
| 10 | [Fault-tolerant and transactional stateful serverless workflows (Beldi)](https://www.usenix.org/conference/osdi20/presentation/zhang-haoran) | OSDI 2020 | durable workflows | Fault-tolerant stateful serverless functions and transactional workflows. | Reuse logging/idempotency principles; do not sell durability as new. | A |
| 11 | [Scalable Atomic Visibility with RAMP Transactions](https://www.bailis.org/papers/ramp-sigmod2014.pdf) | SIGMOD 2014 | selective repair | Metadata exposes partial reads and enables targeted second-round retrieval. | Strong mechanism anchor for dependency metadata plus selective patch. | A |
| 12 | [Noria: Dynamic, Partially-Stateful Data-Flow for High-Performance Web Applications](https://www.usenix.org/conference/osdi18/presentation/gjengset) | OSDI 2018 | partial state | Maintains partial state and reconstructs missing state on demand. | Motivates materializing only the continuation slice needed downstream. | A |
| 13 | [Proof-Carrying Code](https://doi.org/10.1145/263699.263712) | POPL 1997 | evidence validation | A producer attaches evidence that a consumer can validate cheaply. | Call packets evidence-carrying, not proof-carrying, unless formal proofs are implemented. | A |
| 14 | [Safety-Aware Algorithms for Adversarial Contextual Bandit](https://proceedings.mlr.press/v70/sun17a.html) | ICML 2017 | safe online decisions | Contextual-bandit decisions under sequential risk constraints. | Router should explicitly constrain fidelity/effect risk instead of using an unconstrained score. | A |
| 15 | [Automatically Adaptive Conformal Risk Control](https://proceedings.mlr.press/v258/blot25a.html) | AISTATS 2025 | calibrated risk | Adapts risk control to sample difficulty and distributional heterogeneity. | Calibrate fallback/abstention thresholds on train/dev only and report coverage/risk. | A |

`Risk` denotes a direct claim boundary. `A` denotes an adjacent mechanism anchor. It does not denote paper quality or acceptance probability.

## Cross-Domain Mechanism Map

| Neighboring field | Transferable mechanism | AgentRelay adaptation | Non-transferable overclaim |
| --- | --- | --- | --- |
| Continuation-centric OS | capture, pause, migrate, resume | encode a model-independent semantic continuation | “first migratable continuation” |
| Atomic visibility | dependency metadata and targeted repair | detect a non-closed semantic cut and request only missing predecessors | “new distributed transaction protocol” |
| Partial dataflow | materialize state on demand | transmit only the dependency-closed slice required by the next-step obligations | “general partial-state database” |
| Runtime verification | causal visibility and online guards | verify that evidence and effects visible at the target form a legal cut | “first causal verifier for agents” |
| Safe bandits / conformal risk | constrained decision and calibrated abstention | route only when success/fidelity/effect-risk bounds pass; otherwise stay or replay | “formal safety guarantee” without verified assumptions |
| Agent transactions | effect frontier, outbox, compensation, recovery | make frontier status constrain migration and commit mode | “first transactional tool runtime” |

## Closest-Work Boundary

Public-source facts:

- Hera and AgentRouter already study per-step/trajectory-aware heterogeneous model selection.
- Portable Agent Memory studies heterogeneous state transfer with provenance and integrity.
- Atomix and Cordon already study transactional external effects.
- Causal Past Logic already studies causal online verification in distributed LLM-agent workflows.
- Arca already treats portable continuations as an operating-system abstraction.

Project inference, not a verified literature fact:

- The still-open joint question appears to be how to compute a **minimal obligation-closed semantic continuation**, verify and selectively repair it after a heterogeneous model switch, and couple that transfer with calibrated executor and effect-frontier decisions.
- This gap is plausible but not proven exhaustive. A full-text novelty audit must be repeated before manuscript claims are frozen.

## Positioning Rules

- Central object: obligation-closed semantic continuation, not a generic router, serializer, or transaction manager.
- Main mechanism: dependency-closed cut plus evidence metadata and selective patch.
- Controller: joint executor/payload/commit decision with dwell time, measured switch cost, and calibrated abstention.
- Safety scope: migration legality based on effect frontier; Cordon/Atomix-style transaction machinery is acknowledged prior art.
- Evidence standard: demonstrate route reversals and failures that appear only when continuation size, patch probability, and effect progress interact.
- Forbidden claims: first step-level router; first portable state; first continuation; first causal verifier; first transactional agent runtime; formal proof-carrying packet without an actual proof system.

