# Search Notes: Gemma 4-Compatible Agent Benchmarks

Date: 2026-08-06

## Search Questions

1. Which public interactive datasets put Gemma 4 E4B and 12B away from both floor and saturation?
2. Which candidates have executable or deterministic rewards and usable official train/dev/test separation?
3. Which benchmarks exercise actual mid-trajectory handoff state rather than one-shot query routing?
4. Which recent papers make static, CATE, step-level, or execution-verified routing claims unsafe?
5. Which non-neighbor mechanisms can strengthen the controller without becoming a loose bundle?

## Safe Public Queries

- `Gemma 4 E4B 12B tau2 benchmark official model card`
- `tau2-bench official repository gym train test splits user simulator`
- `InterCode official NeurIPS SQL Bash Python Docker execution feedback`
- `Berkeley Function Calling Leaderboard V4 agentic multi turn official`
- `AgentBoard NeurIPS 2024 progress rate easy hard official dataset`
- `TextWorldExpress official train dev test difficulty rooms distractors`
- `ToolSandbox stateful conversational tool benchmark official`
- `CAR-bench official train test hallucination disambiguation`
- `TwinRouterBench step-level execution verified routing benchmark`
- `LLMRouterBench ACL 2026 unified routing benchmark`
- `Meta-Router CATE causal LLM routing ICLR 2026`
- `learning to defer expert ICML PMLR`
- `contextual bandits with knapsacks ICLR 2025`

No attachment text, private prompt, unpublished result row, model credential, or private dataset example was sent to a search engine.

## Sources Checked

- Google DeepMind and Google AI official Gemma 4 model pages/model card
- arXiv primary paper pages and official repositories for recent benchmarks
- NeurIPS, ICLR, ICML/PMLR, ACL Anthology, and OpenReview primary records
- Official benchmark repositories and frozen evaluation versions where available

## Screening Pool

Thirty-four candidates were screened. Fifteen high-value sources were retained in `papers.md`/`papers.csv`. The remaining candidates were excluded or deferred for one or more of these reasons:

- one-shot QA does not exercise handoff state;
- all reported small/open models are near zero;
- evaluation depends primarily on an online LLM judge;
- no usable train/calibration split is available for router learning;
- GUI/browser setup dominates the method experiment;
- the benchmark duplicates WebShop/AppWorld difficulty without a better pair-fit signal;
- labels are locked to another model pool and cannot supervise the Gemma pair directly.

Notable screened-but-not-retained candidates include API-Bank, MINT, MiniWoB++, WebArena, AppWorld, SWE-bench Verified, CompWoB, BabyAI, PDDL, Jericho, ToolBench, AgentBench, BrowserGym, RouteBench, and CodeRouterBench.

## Evidence vs Inference

Verified public facts:

- The Gemma 4 official model card reports Tau2 average scores of 42.2% for E4B and 69.0% for 12B.
- The same card reports LiveCodeBench v6 scores of 52.0% and 72.0%.
- Tau2 exposes text domains, executable task evaluation, a Gym interface, task splits, and an LLM user simulator.
- InterCode uses Docker-backed interactive execution environments.
- BFCL V4 includes agentic, multi-turn, non-live/live, hallucination, and format-sensitivity categories.
- TwinRouterBench already constructs execution-verified per-step downgrade labels.
- Meta-Router already frames routing through CATE.
- LLMRouterBench reports that many sophisticated routers perform similarly to simple baselines under unified evaluation.

Project inferences requiring pilots:

- InterCode-SQL will likely place the Gemma pair in a useful band because of the official coding gap.
- Tau2 will produce paired cloud-rescue and edge-sufficiency rates suitable for the project; aggregate public scores do not prove the paired contingency table.
- Risk-lower-bound, episode-budgeted sequential deferral will yield non-additive gains when coupled to measured continuation tax and effect-frontier legality.

## Version and Reproducibility Notes

- Pin the exact Gemma 4 model revisions, tokenizer/processor, quantization, thinking mode, chat template, and inference library.
- Pin Tau2 to an exact release/commit. The repository had task and grading fixes in 2026, so scores across affected versions are not automatically comparable.
- Freeze the user-simulator model/version and decoding configuration for paired Tau2 comparisons.
- BFCL publishes frozen code/data references; use the official pinned release rather than the moving leaderboard head.
- Do not fit on AgentBoard's released test wrapper. Use the original environment's official training/validation partitions.

## Handoff

- Idea optimization: use learning-to-defer, calibrated lower bounds, and budget shadow pricing as controller ingredients; keep semantic continuation/effect legality as the systems core.
- Experiment design: implement a 20–30 case BFCL preflight, then a 50-task paired Tau2 gate; fall back to InterCode-SQL when a fixed user simulator is unavailable.
- Implementation: add adapters only after the paired gate manifest, version pins, and stop criteria are frozen.
