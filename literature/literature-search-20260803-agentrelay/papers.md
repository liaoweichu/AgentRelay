# Literature Search: Stateful Edge-Cloud Handoff for LLM Agents

Date: 2026-08-03  
Search purpose: novelty grounding, closest-work screening, and benchmark discovery  
Target venue/family: MLSys / systems-oriented ML  
Source-quality policy: primary or stable scholarly sources prioritized; MDPI and untraceable sources excluded

## Summary

- Screened candidates: 26; retained closest/high-value papers: 15.
- The original claim, "first step-level device-cloud routing for long-horizon agents," is covered by **Hera** and substantially overlaps **AgentRouter**, **SWE-Router**, and harness-native agentic routing.
- A defensible opportunity remains at the boundary between routing and systems state: current work normally treats a model switch as a cheap selection event, a full-context replay, or a separate migration problem.
- The recommended paper should study a **stateful handoff protocol** that jointly accounts for switch cost, typed semantic-state fidelity, selective repair, and irreversible-effect safety.
- Strongest routing baselines: Hera, AgentRouter, SWE-Router, RouteLLM, FrugalGPT, and cascade routing.
- Strongest state/continuity baselines: Context Folding, Handoff Debt, Portable Agent Memory, and the two-agent compression study.
- Strongest runtime/safety baseline: Atomix.

## Paper Table

| # | Title | Year | Venue/source | Link | Type | Insight | Completeness | Numeric evidence | Overall | Relevance note |
| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | --- | --- |
| 1 | AgentRouter: Heterogeneous Model Routing for Cost-Optimal Multi-Step Agentic Workflows | 2026 | ICML / PMLR 306 paper PDF | [paper](https://openreview.net/pdf?id=nu3GPfkyJV) | method + benchmark | 4 | 4 | 4 | Risk | Formalizes trajectory-aware per-step model routing and explicitly observes tier-switch frequency; direct novelty threat. |
| 2 | Hera: Learning Long-Horizon Coordination for Device-Cloud Collaborative LLM Agents | 2026 | arXiv preprint | [paper](https://arxiv.org/abs/2605.24598) | pure method | 4 | 4 | 4 | Risk | Directly covers step-level device-cloud routing with imitation learning and cost-aware RL on ALFWorld, WebShop, and AppWorld. |
| 3 | SWE-Router: Routing in Multi-turn Agentic Software Engineering Tasks | 2026 | arXiv preprint | [paper](https://arxiv.org/abs/2607.00053) | method + benchmark | 5 | 4 | 4 | Risk | Uses partial trajectories before escalating and supplies a theoretical value-of-information argument plus trajectory data. |
| 4 | Agentic Routing: The Harness-Native Data Flywheel | 2026 | arXiv technical report | [paper](https://arxiv.org/abs/2607.11399) | system/tool | 4 | 3 | 3 | Risk | Routes at step level from full harness state and turns execution records into router-training data. |
| 5 | Speculative Actions: A Lossless Framework for Faster AI Agents | 2026 | ICLR Oral | [paper](https://openreview.net/forum?id=P0GOk5wslg) | pure method | 5 | 4 | 4 | Risk | Defines action-level speculation with validation and reversible effects; constrains claims around parallelism and rollback safety. |
| 6 | LLM-as-Scheduler: Agentic Workflow Dynamic Scheduling | 2026 | ACL Long Paper | [paper](https://aclanthology.org/2026.acl-long.581/) | pure method | 4 | 4 | 4 | A | Dynamically exits or routes between workflow stages using intermediate artifacts; useful workflow-level baseline. |
| 7 | A Unified Approach to Routing and Cascading for LLMs | 2025 | ICML / PMLR 267 | [paper](https://proceedings.mlr.press/v267/dekoninck25a.html) | theory/proof + method | 5 | 5 | 4 | A | Provides the strongest query-level routing/cascading formulation and identifies quality estimation as the key dependency. |
| 8 | RouteLLM: Learning to Route LLMs with Preference Data | 2025 | ICLR | [paper](https://arxiv.org/abs/2406.18665) | pure method | 4 | 4 | 4 | A | Canonical strong/weak query router and a required coarse-routing baseline. |
| 9 | FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance | 2024 | TMLR | [paper](https://openreview.net/forum?id=cSimKw5p6R) | pure method | 4 | 5 | 4 | A | Canonical LLM cascade; required baseline for escalation-style policies. |
| 10 | Scaling Long-Horizon Agent via Context Folding | 2026 | ICML | [paper](https://openreview.net/forum?id=lNRgWoGfYg) | pure method | 4 | 4 | 4 | Risk | Learns to fold completed sub-trajectories into compact summaries; overlaps generic context-compression claims. |
| 11 | State Compression in Two-Agent LLM Relays: A Closed-World Study of Constraint Preservation | 2026 | arXiv preprint | [paper](https://arxiv.org/abs/2607.18265) | method + benchmark | 3 | 3 | 3 | Risk | Directly tests narrative, structured JSON, and embedding-pruned handoff payloads; narrow setting but close mechanism. |
| 12 | Portable Agent Memory: A Protocol for Cryptographically-Verified Memory Transfer Across Heterogeneous AI Agents | 2026 | arXiv preprint | [paper](https://arxiv.org/abs/2605.11032) | system/tool | 4 | 3 | 3 | Risk | Defines structured, provenance-aware, heterogeneous memory transfer; rules out broad "first portable state" claims. |
| 13 | Handoff Debt: The Rediscovery Cost When Coding Agents Take Over Interrupted Tasks | 2026 | arXiv preprint | [paper](https://arxiv.org/abs/2606.02875) | method + benchmark | 4 | 4 | 4 | Risk | Measures takeover cost under repository-only, raw-trace, summary, and structured-note views; motivates handoff efficiency metrics. |
| 14 | Adaptive AI Agent Placement and Migration in Edge Intelligence Systems | 2025 | arXiv preprint | [paper](https://arxiv.org/abs/2508.03345) | system/tool | 3 | 3 | 3 | Risk | Treats whole-agent placement/migration and transfers essential state; differentiates placement-level from step-switch-level systems. |
| 15 | Atomix: Progress-Aware Transactional Runtime for Agent Tool Calls | 2026 | ICLR Agents in the Wild Workshop | [paper](https://openreview.net/forum?id=UeRbEpSVUz) | system/tool | 4 | 3 | 3 | Risk | Provides transactional effect visibility, compensation, and progress-gated commit; direct baseline for commit/rollback semantics. |

Scores assess source-reported insight, completeness, and numerical evidence, not acceptance probability or the viability of AgentRelay.

## Closest-Work Clusters

### Cluster 1: Step-Level And Trajectory-Aware Routing

- Representative papers: AgentRouter, Hera, SWE-Router, Agentic Routing.
- What this cluster already solves: per-step strong/weak model selection, trajectory-conditioned decisions, cloud-usage objectives, and partial-trajectory escalation.
- Remaining gap: switch execution is usually abstracted as a model choice; state serialization, target-model rehydration fidelity, repair traffic, irreversible side effects, and measured switch latency are not jointly optimized.
- Differentiation route: make the **handoff transaction** rather than the router classifier the central object.
- Effect on this project: prohibit "first step-level router" and "critical steps are sparse" as novelty claims; both are prior-art territory.

### Cluster 2: Query Routing, Cascades, And Dynamic Workflows

- Representative papers: RouteLLM, FrugalGPT, cascade routing, LLM-as-Scheduler.
- What this cluster already solves: cost-quality model selection, escalation, early exit, and adaptive workflow truncation.
- Remaining gap: query-level quality estimators do not model cross-model state compatibility or tool-effect consistency across a long trajectory.
- Differentiation route: compare against these methods, but use switch-aware objectives and state-fidelity constraints unavailable to query routers.
- Effect on this project: a classifier-only contribution is insufficient.

### Cluster 3: Context Compression And Handoff Continuity

- Representative papers: Context Folding, State Compression in Two-Agent LLM Relays, Portable Agent Memory, Handoff Debt.
- What this cluster already solves: summary-based context reduction, structured payloads, heterogeneous memory portability, and takeover-effort measurement.
- Remaining gap: no retained work jointly chooses **when to switch, what minimal state delta to send, whether the target recovered required invariants, and what to patch on failure** under a deployment budget.
- Differentiation route: typed state invariants plus deterministic/learned fidelity checks and selective patching, evaluated across multiple public agent benchmarks and heterogeneous model pairs.
- Effect on this project: "JSON is better than summaries" is too weak; verification and adaptive repair must be central.

### Cluster 4: Migration, Transactions, And Effects

- Representative papers: Adaptive AI Agent Placement and Migration; Atomix; Speculative Actions.
- What this cluster already solves: placement-level agent migration, progress-gated external effects, and reversible speculative actions.
- Remaining gap: model-routing papers ignore effect boundaries, while transaction papers do not optimize heterogeneous model selection and semantic state transfer.
- Differentiation route: introduce commit classes (`read_only`, `reversible`, `irreversible`) into the handoff cost/risk model and require an idempotency ledger for retries.
- Effect on this project: transactional safety should be an integrated systems mechanism, not an unrelated safety add-on.

## Opportunity Map

| Cluster | Status | Open gap | Possible direction | Evidence needed | Risk |
| --- | --- | --- | --- | --- | --- |
| Step-level routing | covered central claim | Switch is treated as cheap/seamless | Switch-cost-aware sequential routing | Show routing reversals when measured transfer cost is included | High |
| Semantic handoff | crowded but open | Compression lacks invariant-level recovery guarantees | Typed semantic delta with verify-and-patch | Fidelity, task success, bytes/tokens, repair rate across model pairs | Medium |
| Stateful migration | deployment/system gap | Whole-agent migration and step routing are disconnected | Joint decision over executor and payload | End-to-end latency/quality Pareto on real public tasks | Medium |
| External effects | deployment/system gap | Route changes can duplicate or invalidate side effects | Commit-aware handoff ledger | Controlled failure/retry experiments with official benchmark tools | Medium |
| Online adaptation | crowded but open | Static policies poorly reflect changing transfer/latency cost | Lightweight constrained contextual bandit | Regret/constraint violations under recorded, non-synthetic deployment traces | Medium-high |

## Benchmark And Dataset Candidates

| Name | Link | Task | Metrics | Baselines | Fit | Risks |
| --- | --- | --- | --- | --- | --- | --- |
| ALFWorld | [official paper](https://openreview.net/forum?id=0IOX0YcCdTn), [HF raw mirror](https://huggingface.co/datasets/awawa-agi/alfworld-raw) | Stateful text-world interaction | success, steps, invalid actions, transfer tokens, latency | device-only, cloud-only, RouteLLM, FrugalGPT, Hera-style router | High; used by Hera and supports official train/seen/unseen splits | HF copy is a mirror; pin upstream version and verify hashes/licenses |
| WebShop | [official paper](https://proceedings.neurips.cc/paper_files/paper/2022/hash/82ad13ec01f9fe44c01cb91814fd7b8c-Abstract-Conference.html) | Grounded web-shopping agent | task reward, success, steps, token/transfer cost | same routing baselines | High; used by Hera | Environment setup and search corpus are heavier than static HF data |
| AppWorld | [official project](https://github.com/StonyBrookNLP/appworld) | Multi-app API interaction | task and scenario goal completion, collateral effects, latency | cloud/device/routing/transaction baselines | Very high for state and side-effect semantics | Some tasks involve simulated users; use the official public benchmark unmodified |
| AgentProcessBench | [HF dataset](https://huggingface.co/datasets/LulaCola/AgentProcessBench) | Step-level process quality across BFCL, GAIA-dev, HotpotQA, and tau2 | process labels, step accuracy, calibration | criticality/value estimators | High for local pilot and estimator diagnostics | Public traces are evaluation material; do not leak labels into downstream test prompts |
| GAIA | [HF dataset](https://huggingface.co/datasets/gaia-benchmark/GAIA) | General tool-augmented autonomy | official answer accuracy and level breakdown | model-only and routing variants | Medium; useful transfer test | Gated access and private test answers require official evaluation discipline |

## Citation And Positioning Cautions

- Do not claim the first step-level, trajectory-aware, device-cloud, or harness-native agent router.
- Do not claim the first structured or cryptographically verifiable heterogeneous agent-memory transfer protocol.
- Do not claim the first transactional tool-call runtime or first rollback-capable agent runtime.
- The paper can claim a new **joint problem and protocol** only if experiments isolate all four components: switch-aware routing, typed delta, verification/patch, and commit-aware execution.
- Preprint/workshop claims should be cited with their exact source status rather than described as archival conference results.

