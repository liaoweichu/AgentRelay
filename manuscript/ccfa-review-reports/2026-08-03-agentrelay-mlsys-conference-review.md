# AgentRelay 全量会议审稿报告

## 1. Report Metadata

- Review date: 2026-08-03
- Target venue/year/track: MLSys 2027，主会研究论文，暂定
- Paper title: AgentRelay: Switch-Cost-Aware and Transaction-Safe State Handoffs for Edge-Cloud LLM Agents
- Input materials reviewed: manuscript/main.tex、manuscript/references.bib、ccfa.yaml、src/agentrelay、tests/test_core.py、实验计划、方法规范、执行指南、配置模板与文献检索档案
- Search basis: 既有 26 篇筛选池，以及本轮对 agent handoff、edge-cloud state transfer、verified patch、transactional tool effects 的公开安全检索
- Report file: manuscript/ccfa-review-reports/2026-08-03-agentrelay-mlsys-conference-review.md
- Reviewer mode: full scientific + writing + LaTeX/source audit
- Privacy boundary: 未将附件原句或未发表正文提交给搜索引擎；只使用公开概念词检索

## 2. Desk Rejection Assessment

- Paper length — uncertain。源文件约 1,000 行、正文近似 5,800 词，含 5 个主结果表和长附录；本机无 LaTeX 引擎，无法得到页数。MLSys 2027 规则尚未公开检索到；MLSys 2025 官方规则为双栏正文 10 页，不含参考文献。必须在 2027 官方模板发布后重新验证。
- Topic compatibility — pass。问题位于 ML 系统、agent runtime、edge-cloud inference 和可靠执行的交叉区域，符合 MLSys 的系统导向。
- Minimum quality — fail as a submission，pass as a pre-experimental research package。方法、边界、实验和复现计划完整，但所有主结果仍为 TBD，不能作为完成论文送审。
- Policy/anonymity/compliance — uncertain。作者匿名，未发现身份信息；但当前是 venue-neutral/NeurIPS 回退样式，不是 MLSys 2027 官方模板，且正式 revision 尚未锁定。
- Prompt injection and hidden manipulation detection — pass。预实验警示框是诚信声明，不是面向审稿模型的操纵指令；未发现隐藏指令或诱导评分文本。
- Ethics and reviewability — pass with one unresolved item。数据、隐私、真实外部效应边界和不可判定重试均有披露；最终版本还需报告 GPU 时长和能耗或功耗代理。

Desk rejection risk: likely if submitted now  
Reason: 主结果完全缺失，且正文明确标注 internal pre-experimental draft。  
Can be fixed before review: yes，但依赖完整实现和正式实验，不是文字润色可解决。

## 3. Paper Summary And Contribution Map

论文把 edge-cloud agent 的模型切换重新定义为有成本、有状态且可能产生外部效应的 handoff transaction。系统对每一步联合选择 executor、transfer mode 和 commit mode；状态通过类型化、内容寻址的 full/delta packet 传递，接收端验证 identity、parent、provenance、invariant 和 effect history，缺失时请求窄 patch；effect ledger 用 canonical key 和状态转换约束重试；路由器显式计入编码、通信、rehydration、验证和 patch 的 relay tax。

- Claimed problem: 模型路由器把有状态切换近似为无状态模型选择，忽略真实交接成本和外部效应一致性。
- Claimed gap: 现有路由、状态压缩、迁移和 transactional runtime 分别处理问题的一部分，但没有联合选择执行器、payload 和 effect mode。
- Contribution C1: measured relay tax 与 switch-cost reversal。
- Contribution C2: executor、payload、commit 的联合受约束决策。
- Contribution C3: typed delta、deterministic verification 与 patch-on-demand。
- Contribution C4: handoff-aware effect ledger 与 failure recovery。
- Evidence package: 目前只有代数区间、13 个软件单元测试和完整实验注册；没有公共 benchmark native-inference 结果。
- Stated limitations: benchmark、exactly-once 边界、state-fidelity proxy、单卡硬件现实性、模型依赖、instrumentation、security/privacy 均有明确披露。

## 4. Search And Related-Work Basis

Queries used:

- LLM agent state handoff routing edge cloud context transfer
- LLM agent transactional tool calls exactly once idempotency handoff
- schema grounded state mutation reliable auditable LLM agents
- verified structured patches shared-state LLM workflows
- edge LLM handover KV cache transfer prefill

Sources searched: arXiv、OpenReview、PMLR、官方 MLSys 页面和项目主页。  
Source-quality screening status: 只用论文页面、正式 proceedings 或官方 conference 页面作为结论依据；二手页面仅用于发现。

Closest works already covered:

- Hera 与 AgentRouter：step/trajectory-level heterogeneous model routing。
- Context Folding、Portable Agent Memory、Handoff Debt：状态压缩和接管成本。
- Atomix、Speculative Actions：progress barrier、rollback 和 effect safety。

Newly found direct risks:

- [Cordon: Semantic Transactions for Tool-Using LLM Agents](https://arxiv.org/abs/2606.17573) 定义 task-scoped semantic transaction、effect outbox、staging、validation、commit 和 recovery，直接压缩 C4 的独立新颖性。
- [PatchBoard: Schema-Grounded State Mutation for Reliable and Auditable LLM Multi-Agent Collaboration](https://arxiv.org/abs/2605.29313) 使用 schema、JSON Patch、deterministic invariant kernel 和 transactional commit，直接压缩 C3。
- [PatchOptic](https://arxiv.org/abs/2607.05483) 使用 projected views 与 verified structured patches，进一步要求本文说明 read-view、write-region 与 patch contract 的区别。
- [QKVShare](https://arxiv.org/abs/2605.03884) 研究多 agent 的 quantized KV-cache handoff；虽然它是 latent same-model state，不等于跨模型 semantic state，但必须进入 handoff 表征版图。
- [Low-Latency Edge LLM Handover](https://arxiv.org/abs/2603.28018) 联合选择 token prefill 和 KV transfer 以降低 handover delay，要求 C1/C2 明确区别 semantic trajectory transfer 与 same-model serving handover。
- [Learning to Hand Off](https://arxiv.org/abs/2605.19140) 形式化 interface-constrained handoff learning，要求本文避免把 joint handoff decision 描述为全新抽象。

Unverified related-work risks:

- SmartGen 等 2026 年 7 月末新出现的 selective KV transfer 工作仍需在投稿前做一次 literature-monitor 更新。
- Cordon、PatchBoard 和 PatchOptic 的官方代码可运行性尚未核实，不能先承诺可执行基线。

## 5. Expected Review Outcome

Expected outcome: reject in current submission state；作为项目 proposal 属于 above-average、证据门设计严谨。  
Main accept signal: 真实系统问题明确，四条主张有互不替代的实验，复现和反泄漏设计显著强于普通早期草稿。  
Main reject signal: 论文没有任何正式结果，核心 runtime 尚不能跑通一个官方 benchmark；C3/C4 又遭遇新近直接 prior art。  
Confidence: 4/5。完整 TeX、代码和附录可检查，但缺 PDF、正式环境、锁定资产和实验记录。

## 6. Strengths And Weaknesses

### Strengths

- Introduction 第 4--5 段把 prompt routing 与 stateful transaction 的差异讲得清楚，贡献列表具体且可证伪。
- Section 2 的 reversal interval 简洁地说明为什么 measured handoff cost 可能改变决策，适合作为 C1 的分析入口。
- Sections 5.2--5.6 对真实公开数据、官方 split、native inference、公平基线、paired statistics 的约束完整。
- Section 8 主动界定 exactly-once 不可能性、单卡 profile 与真实 edge-cloud 部署差异，避免广义保证。
- 配置层 fail-closed：正式配置禁止 sample limit、要求 immutable revision、禁止 test-label access，并固定 AutoDL 持久目录。
- 13 个软件测试验证 packet round trip、tamper detection、patch、effect conflict、route reversal、trace exhaustion 和统计函数；这些是合法的软件证据，且没有被冒充为论文实验。

### Major and fatal weaknesses

Weakness: 所有主张缺少正式实验结果。  
Evidence basis: Section 6 明示 No formal result exists；五张结果表全部是 TBD；ccfa.yaml 中 C1--C4 均为 planned。  
Reviewer deduction: Evidence 1/5；这是独立致拒理由。  
Required fix: 在锁定公开资产后完成 E1--E4，保留原始 JSONL、manifest、paired tests 和失败记录，并由机器生成表格。

Weakness: 系统实现尚不是可端到端运行的 research prototype。  
Evidence basis: PublicBenchmarkAdapter 只有抽象接口；仓库没有 ALFWorld/WebShop/AppWorld adapter、trajectory runner、tool interceptor、baseline runner 或 failure injector。  
Reviewer deduction: 实现章节比实际 artifact 多承诺一层；Reproducibility 只能给 3/5。  
Required fix: 至少先打通 ALFWorld train/debug 的 reset-step-packet-infer-evaluate 闭环，再打通 AppWorld effect path，随后扩展 WebShop。

Weakness: 论文与代码的策略目标不一致。  
Evidence basis: Section 2/4.6 声称在 success/fidelity 约束内最小化 measured cost；policy.py 实际在可行集合中最大化 success_weight × predicted_success − cost。  
Reviewer deduction: C1/C2 的 deployed objective 无法从正文复现。  
Required fix: 二选一并固定：实现真正的 constraint-first minimum-cost policy，或把 scalarized utility 写成部署规则，并在验证集预注册 success_weight。

Weakness: fail-closed 和 exactly-once 的关键状态转换尚未由代码实现。  
Evidence basis: HandoffCoordinator 在 final_validation.invalid 时仍返回 transaction；EffectLedger.commit 对已 committed 记录可再次 commit；compensate 没有限制必须从 committed 进入；ledger 未连接真实 tool adapter。  
Reviewer deduction: C3/C4 的核心保证当前不成立，Soundness 降为 2--3 区间。  
Required fix: 强制非法最终状态抛出/abort；编码显式状态机；实现 before-send 到 retry 的稳定 effect key；加入所有非法转换和 uncertain acknowledgement 测试。

Weakness: C3/C4 的相关工作定位已过时。  
Evidence basis: Related Work 未包含 Cordon、PatchBoard、PatchOptic、QKVShare 和 edge LLM handover。  
Reviewer deduction: 目前四个创新点可能被看作 Hera + PatchBoard + Cordon 的组合工程。  
Required fix: 增加 technical-axis comparison table；把 novelty 收敛为 joint cross-model routing transaction、measured routing reversal 和机制交互，而不是分别声称 typed patch 或 transaction 首创。

### Writing and presentation weaknesses

- 缺少系统总览图。Section 3 的 packet、validator、policy、ledger 很清楚，但 reviewer 需要一张从 benchmark harness 到 source packet、policy、transport、receiver、tool boundary 的单页图。
- Section 4 声称 benchmark adapters 和 immutable native artifacts 已实现，但实际只有接口；措辞应改成 core 已实现、adapters pending，直到代码闭环。
- Effect class 和 commit mode 在正文与代码不完全一致。正文描述 idempotent、read、prepare-commit 等操作模式，代码枚举为 read_only/reversible/irreversible/unknown 与 immediate/barrier/compensating。
- 主表用 resizebox 缩放八列，正式双栏中可能不可读。应拆成 quality/efficiency 两张表或使用跨栏表并减少列。
- 结果段目前只有 planned interpretation。实验完成后每张表需要一段 orienting result paragraph，先给 effect size 和 interval，再解释机制。

## 7. Potentially Missing Related Work

Work: Cordon  
Status: searched  
Why relevant: task-scoped semantic transactions 和 irreversible effect staging 与 C4 高度重叠。  
Overlap: outbox、validation、commit、rollback/recovery、audit metadata。  
Needed comparison: technical-axis contrast；若代码可用，在 AppWorld E4 中加入或解释为什么 effect semantics 不可兼容。

Work: PatchBoard  
Status: searched  
Why relevant: schema-grounded JSON Patch、deterministic invariants、transactional mutation 与 C3 高度重叠。  
Overlap: typed patch、validation、auditability。  
Needed comparison: 在 E2 加 PatchBoard-style shared-state baseline 或 faithful structured-patch baseline；突出跨 heterogeneous executor 的 parent acknowledgement、selective patch 和 routing coupling。

Work: PatchOptic  
Status: searched  
Why relevant: projected view 和 verified structured update 是 C3 的最近机制族。  
Overlap: path-level read/write contract 和 validation。  
Needed comparison: 说明 AgentRelay 是否定义 authorized write region；若没有，接受能力边界而不是泛化可靠 shared state。

Work: QKVShare and Low-Latency Edge LLM Handover  
Status: searched  
Why relevant: 直接研究 handoff payload、prefill 与通信成本。  
Overlap: handoff latency、transfer representation 和 receiver rehydration。  
Needed comparison: 明确 latent same-model KV handoff 与 model-neutral semantic state 的适用条件；在 relay-tax 分解中保留 prefill baseline，但不要把无法跨模型的 KV 方法作为不公平主基线。

Work: Learning to Hand Off  
Status: searched  
Why relevant: handoff interface constraints 和 joint workflow learning 的理论化。  
Overlap: handoff epoch、interface representation gap。  
Needed comparison: 说明本文不提供收敛理论，贡献在 measured cross-model system transaction。

## 8. Claim-Evidence Audit

| Claim | Where stated | Evidence provided | Strength | Reviewer deduction | Required fix |
| --- | --- | --- | --- | --- | --- |
| C1: relay tax can reverse routes | Abstract, Introduction C1, Eq. 4, Conclusion | 代数 interval；1 个软件 fixture test | weak | 说明可能性，但没有证明真实 trajectory 中存在或有益 | E3 至少两个公开 benchmark 的 component measurement、counterfactual disagreement 和 native continuation |
| C2: joint action improves Pareto frontier | Introduction C2, Section 3.6, Conclusion | 只有 action formulation 和计划表 | absent | 中心性能主张不可评分 | E1 完整 split；与 executor-only、fixed-full、fixed-delta、Hera-style 的同协议 frontier |
| C3: verified typed delta reduces context while preserving fidelity | Abstract, Introduction C3, Sections 3.2--3.4 | packet round-trip 和 missing-trace patch 单测 | weak | 软件可行，不代表跨模型 continuation fidelity；与 PatchBoard 重叠 | E2 matched handoff positions、size match、action agreement、continuation success、patch/fallback diagnostics |
| C4: ledger prevents duplicate/invalid effects | Abstract, Introduction C4, Section 3.5 | authorize-after-commit 和 scope-conflict 单测 | weak | 直接 commit 状态机仍有漏洞，且无真实 adapter/failure injection | 修复状态机；AppWorld E4 四个 interruption point；与 Cordon/strong barrier 比较 |

## 9. Experiment / Benchmark / Reproducibility Audit

### Baselines

计划覆盖 edge/cloud anchors、FrugalGPT、RouteLLM、AgentRouter、Hera、state-transfer variants 和 Atomix-inspired barrier，方向正确。缺失 Cordon 与 PatchBoard/PatchOptic-style closest baselines。对无法运行的最新工作应提供结构化差异和明确实现状态，不能用不匹配的论文数字填主表。

### Ablations

已有 relay tax、executor-only、fixed delta、no validation、verify-no-patch、no ledger 等机制消融。还需一个 interaction ablation：joint policy + verified delta 但无 ledger，以及 joint policy + ledger 但 full replay，用于证明四部分不是松散拼装。

### Datasets and benchmarks

ALFWorld、WebShop、AppWorld 与 AgentProcessBench 的角色分离清楚，split discipline 良好。风险在于 formal config 的 Hugging Face datasets 列表只直接包含 AgentProcessBench 和 ALFWorld mirror；WebShop/AppWorld 依赖 repository checkout，但尚无资产 hash、adapter 或官方 evaluator smoke record。

### Metrics and statistics

质量、latency、transfer、fidelity、effect correctness 和 component tax 覆盖完整。Paired bootstrap、exact McNemar 和 Holm 已有软件测试。需预注册 bootstrap unit 是 task 还是 run，三次 stochastic run 与 task-level pair 如何层级聚合，以及 Pareto nondomination 的不确定性展示。

### Robustness and failure cases

记录式网络 trace、patch storm、stale parent、unknown tool、cross-family 和四个 interruption points 设计较强。当前没有 trace asset，也没有 failure injector。真实 edge 与 cloud 没有物理分离，必须继续把 live、service-profile replay 和 trace replay 分开。

### Implementation and artifacts

Core package 有良好模块化与 fail-closed config，但缺少 end-to-end orchestration。当前 locked configs、下载资产、native rollouts、router artifact、run manifests 和 result summaries 均不存在。local RTX 4080 只确认到 GPU 可见与软件单测；这不能替代 E0。

### Limitations

正文对 sandbox、exactly-once impossibility、hardware realism 和 privacy 的边界诚实。还应增加 prior-art-driven limitation：typed JSON patch 与 transactional staging 不是单独的新概念，本文成功与否取决于它们和 routing 的可测交互。

## 10. Multi-Reviewer Panel

Reviewer: Best-justified reviewer  
Expertise: ML systems and agent runtimes  
Likely score: 5/10 if evaluated as a proposal；3/10 as a submission  
Confidence: 4/5  
Main positive signal: 统一问题定义能把路由、传输、验证和 effect correctness 放到同一个可测协议中。  
Main negative signal: 没有真实结果，无法判断 integration 是否产生新知识。  
Evidence basis: Sections 2--5 与完整 claim-evidence contract。  
Score-change condition: E1--E4 完整且至少 C1/C3 或 C1/C4 出现机制级正结果，可上调 2--3 分。

Reviewer: Critical reviewer  
Expertise: systems novelty  
Likely score: 2/10  
Confidence: 4/5  
Main positive signal: 诚实标记 TBD，未伪造数字。  
Main negative signal: Hera + PatchBoard + Cordon 可能覆盖三个主要模块，剩余只是组合。  
Evidence basis: 新检索 closest works 与 Related Work 缺口。  
Fatal concern if any: 若无法显示 routing decision reversal 和跨模块 interaction，novelty collapse。  
Score-change condition: closest-work axis table、strong baselines、validated reversal 和 interaction ablations。

Reviewer: Method / soundness  
Expertise: distributed state and transactions  
Likely score: 3/10  
Confidence: 5/5  
Main positive signal: packet identity、parent hash、patch 和 uncertainty boundary 定义合理。  
Main negative signal: policy objective、validation abort 和 ledger transition 与论文不一致。  
Evidence basis: policy.py、runtime.py、effects.py 对照 Sections 2、3.4--3.6。  
Fatal concern if any: 非法 state 仍可能被返回给上层，重复 direct commit 未被拒绝。  
Score-change condition: 修复状态机并增加 transition-complete property tests 和 end-to-end tool test。

Reviewer: Evidence / experiment  
Expertise: agent benchmark evaluation  
Likely score: 2/10  
Confidence: 5/5  
Main positive signal: E1--E7 和指标设计覆盖面强。  
Main negative signal: evidence package 目前为空。  
Evidence basis: 所有结果表 TBD，formal runs pending。  
Score-change condition: 完整 split、同 seed paired execution、raw manifests 与 uncertainty tests。

Reviewer: Novelty / positioning  
Expertise: agent routing and state transfer  
Likely score: 3/10  
Confidence: 4/5  
Main positive signal: 不再声称 first step-level router。  
Main negative signal: 新近直接近邻未覆盖，C3/C4 独立表述过强。  
Evidence basis: Section 2 与公开检索。  
Score-change condition: 将核心 novelty 改为 joint transaction and measured interaction，并完成 closest baselines。

Reviewer: Writing / clarity  
Expertise: systems paper presentation  
Likely score: 4/5 on clarity  
Confidence: 4/5  
Main positive signal: problem-gap-insight-contribution 顺序稳定，所有未来时态和 TBD 清楚。  
Main negative signal: 无系统图，Implementation 的完成度措辞超过代码。  
Evidence basis: Introduction、Sections 3--4。  
Score-change condition: 加总览图，修订 implemented/planned 边界，实验完成后添加 result-led paragraphs。

Reviewer: Ethics / reproducibility  
Expertise: artifact integrity  
Likely score: 4/5  
Confidence: 4/5  
Main positive signal: public-only、no leakage、immutable manifests、native inference 与 sandbox 限制明确。  
Main negative signal: 没有锁定 revision、license snapshot、GPU-hour record 或可执行 run。  
Evidence basis: Section 9、configs 和 scripts。  
Score-change condition: 锁定资产并产出可重跑 manifest 与 compute report。

Reviewer: Domain application  
Expertise: edge-cloud deployment  
Likely score: 3/5  
Confidence: 4/5  
Main positive signal: relay tax 把通信和 rehydration 显式化。  
Main negative signal: 单卡 sequential profile 加 trace replay 不等于真实 edge-cloud service。  
Evidence basis: Sections 5.3 和 8。  
Score-change condition: 至少增加双进程/双设备 loopback 或远端 RTT sensitivity，并保持主张限定。

Reviewer: Evidence / ablation  
Expertise: causal mechanism isolation  
Likely score: 2/5  
Confidence: 5/5  
Main positive signal: 单模块消融表已预注册。  
Main negative signal: 没有 interaction ablation，也没有数据。  
Evidence basis: Table 7 和 experiments/experiment-plan.md。  
Score-change condition: 添加两两 interaction variants 和机制指标。

Reviewer: Reproducibility  
Expertise: open-source systems artifacts  
Likely score: 3/5  
Confidence: 5/5  
Main positive signal: core 无重依赖、配置验证和统计单测易审计。  
Main negative signal: benchmark adapter、runner、baseline、failure injection 缺失。  
Evidence basis: repository file inventory。  
Score-change condition: 一条命令完成 locked preflight、一个 public task native rollout、aggregate 和 manifest verification。

Reviewer: Novice advocate  
Expertise: first-pass readability  
Likely score: 4/5  
Confidence: 4/5  
Main positive signal: relay tax 和 transaction 的直觉容易恢复。  
Main negative signal: typed packet、delta、effect ledger 和 policy 的关系只能靠长文本拼接。  
Evidence basis: Section 3 没有 architecture figure。  
Score-change condition: 一张编号流图显示 route、transfer、verify/patch、commit 的时间顺序。

Reviewer: AC / meta-reviewer  
Expertise: MLSys synthesis  
Likely score: 3/10  
Confidence: 4/5  
Main positive signal: 若实验证明 handoff cost 改变最优 route，这会是清晰的 ML systems insight。  
Main negative signal: 当前是优秀研究计划而非完成论文，且 novelty 基线刚被新工作抬高。  
Evidence basis: 全稿、代码和本轮 related-work search。  
Fatal concern if any: evidence absence。  
Score-change condition: 先关闭 soundness gaps，再以 E0/E2/E4 gates 决定是否值得跑全量 E1。

Agreement: 研究问题重要、稿件诚实、实验设计成熟，但当前不可提交。  
Disagreement: sympathetic reviewer认为联合 formulation 可能足够；critical reviewer认为若无 interaction evidence 就只是 known-component integration。  
Decisive positive axis: measured routing reversal plus transaction correctness on real public trajectories。  
Decisive negative axis: zero empirical evidence and incomplete end-to-end runtime。  
Unresolved evidence: locked assets、real handoff profile、benchmark fidelity、effect failure injection。  
AC stance: reject now；完成 P0 gates 后重新审稿，而不是先润色。

## 11. Concerns Table

| ID | Severity | Concern | Evidence basis | Affected criterion | Fix class | Required action | Owner skill | Score-change condition |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | fatal | 无正式结果 | Section 6 全 TBD | Evidence | experiment | 完成 E1--E4 并机械生成结果 | ccf-experiment-designer | Evidence 至少升至 3 |
| R2 | fatal | 无官方 benchmark 端到端 runner | 仅 abstract adapter | Soundness/Reproducibility | method/soundness | 实现 ALFWorld、AppWorld、WebShop adapters 与 runner | ccf-pipeline-orchestrator | 可复现 native episode |
| R3 | major | policy objective 与论文不一致 | policy.py 对照 Sections 2/3.6 | Soundness | method/soundness | 固定 constraint-first 或 scalarized 规则 | ccf-paper-writer + implementation | 同一配置可复现 route |
| R4 | major | validation/effect 状态机不满足承诺 | runtime.py、effects.py | Soundness | method/soundness | abort invalid；完整非法转换测试 | ccf-integrity-auditor + implementation | C3/C4 软件门通过 |
| R5 | major | Cordon/PatchBoard 等缺失 | Section 2 | Novelty | related-work | 更新近邻、差异表和基线 | ccf-literature-searcher | Originality 可稳定到 3--4 |
| R6 | major | 组件组合缺 interaction evidence | ablation plan | Originality/Evidence | experiment | 增加 joint-by-component 交互消融 | ccf-experiment-designer | 排除 loose integration 批评 |
| R7 | moderate | 单卡 profile 的部署现实性有限 | Sections 5.3/8 | Significance | accepted-limitation | 分离 live/profile/trace 并增加 RTT sensitivity | ccf-experiment-designer | 不再产生绝对 latency 误读 |
| R8 | moderate | effect taxonomy 文码不一致 | Section 3.5 与 schema.py | Clarity/Soundness | writing | 统一 effect class、commit mode 和 state diagram | ccf-paper-writer | 读者可一一映射代码 |
| R9 | moderate | 缺少系统图 | Section 3 | Clarity | figure/table | 生成 architecture + sequence figure | ccf-visual-composer | Clarity 4 升 5 |
| R10 | moderate | MLSys 2027 模板和页限未知 | preamble 与无 PDF | Format | venue-mismatch | 官方规则发布后迁移和编译 | ccf-submission-checker | desk risk 降为 low |
| R11 | minor | 主表可能缩放不可读 | 多个 resizebox | Clarity | figure/table | 拆表或精简列 | ccf-visual-composer | 双栏 PDF 可读 |

## 12. AC / Meta-Review

Reviewer consensus: 当前 artifact 展示了严谨问题分解与诚信控制，但没有提交级证据。  
Reviewer disagreement: 联合 formulation 是否构成足够 originality，要由 E3 routing reversal 和 interaction ablations 决定。  
Decisive acceptance axis: 在完整公共 benchmark 上证明 switch tax 不是 bookkeeping，而会改变可验证的最优执行路径；同时 state/effect protocol 在故障下提供可测正确性。  
Decisive rejection axis: 即使未来成功率不错，若没有 PatchBoard/Cordon/Hera 级强基线和机制交互，仍会被判为 known-component assembly。  
AC stance: current reject，P0 gates 后重审。  
Discussion risks: 新工作快速出现；exactly-once 用词；单卡模拟 edge-cloud；baseline reproduction status。

## 13. Quantitative Scores

| Dimension | Score (1-5) | Confidence (1-5) | Evidence basis | Deduction / score-change condition |
| --- | ---: | ---: | --- | --- |
| Novelty | 3 | 4 | 四点贡献与新近 closest work | C3/C4 单独新颖性弱；joint interaction evidence 可升至 4 |
| Soundness | 2 | 5 | policy/runtime/effects 代码审计 | 三个可复现语义差异；状态机修复和端到端 test 后可升至 3--4 |
| Evidence | 1 | 5 | 所有主结果 TBD | E1--E4 完整且可核验后才可升分 |
| Significance | 4 | 4 | edge-cloud agent state/effect bottleneck | 真实 reversal 和 deployment sensitivity 决定能否保持 4 |
| Clarity | 4 | 4 | problem-gap-contribution 明确 | 系统图、taxonomy 对齐、结果叙事可升至 5 |
| Reproducibility | 3 | 5 | core tests/config/scripts；无 adapters/runs | locked one-command pipeline 后可升至 4--5 |
| Ethics / Limitations | 4 | 4 | Sections 8--9 | 加 compute/energy 与 licenses report 可升至 5 |

Quality: 2/5  
Clarity: 4/5  
Significance: 4/5  
Originality: 3/5  
Soundness: 2/5  
Evidence: 1/5  
Reproducibility: 3/5  
Ethics / Limitations: 4/5  
Overall: 3/10  
Confidence: 4/5  
Recommendation: reject in current state  
Verdict: 修复核心状态机、实现两个 P0 benchmark 路径并得到真实 P0 证据可上调约 1 分；完整 E1--E4 出现正向且非支配的结果可再上调 2--3 分。若 C1 无 route reversal 且 C3/C4 被 closest work 等价覆盖，则应降为 2 或转为窄化负结果论文。

### Writing scorecard

| Dimension | Weight | Score | Confidence | Evidence basis | Concrete repair |
| --- | ---: | ---: | ---: | --- | --- |
| Storyline and motivation | 12 | 4 | 4 | Introduction | 保持当前主线 |
| Contribution display | 12 | 4 | 4 | 四点列表 | 改成 joint system insight + mechanisms |
| Paragraph logic | 10 | 4 | 4 | 全稿 | 压缩重复限制语句 |
| Claim-evidence alignment | 14 | 3 | 5 | TBD 透明但无证据 | 实验后每项 claim 指向表/图 |
| Method readability | 10 | 3 | 4 | Section 3 无图 | 加 architecture/sequence figure |
| Experiment narration | 10 | 2 | 5 | Results 只有模板 | 每表添加结果导向段落 |
| Related-work positioning | 8 | 2 | 5 | 缺直接 2026 works | 增加 technical-axis comparison |
| Terminology consistency | 8 | 3 | 5 | effect taxonomy 文码不一致 | 统一枚举与状态图 |
| LaTeX and format discipline | 8 | 2 | 4 | 非 MLSys 模板、未编译 | 迁移官方模板并编译 |
| Reviewer-facing risk | 8 | 2 | 5 | evidence/implementation gap | 关闭 R1--R5 |

Weighted writing score: 3.0/5  
Writing risk band: high，主要来自未完成证据与格式，而不是语言可读性。

## 14. Questions For Authors

1. 部署策略到底是在质量约束内最小成本，还是最大化 success-weighted utility？success weight 如何只用 validation 选择？
2. 当 final validation 失败时，上层 runtime 的唯一合法行为是什么，代码在哪里阻止模型继续生成？
3. effect key 在 send 后 environment version 变化、ack 丢失时如何保持稳定，而不是重新计算出新 key？
4. Cordon 与 PatchBoard 的哪一个具体机制在 AgentRelay 中被改进，而不只是连接到 router？
5. 若 E3 找不到实际 beneficial reversal，论文会保留哪一个主贡献，标题是否仍强调 switch-cost-aware？
6. WebShop 的 action 是否足以支持 C4，还是 C4 严格只在 AppWorld；主表如何避免把 read-only 任务的零 duplicate 当成安全证据？
7. 一个完整 official task 从 reset 到 aggregate 的可复现命令是什么？

## 15. Score Revision Criteria

Raising the score would require:

- 关闭 R3/R4 的语义差异并扩展状态机测试；
- 端到端跑通锁定的 ALFWorld 与 AppWorld public tasks；
- 用 Cordon、PatchBoard、Hera 形成公平的 closest-work evidence；
- E3 证明至少一类真实 route reversal，E2/E4 证明 fidelity/effect 机制；
- 完整 E1--E4、paired uncertainty 和 failure disclosure。

Lowering the score would be triggered by:

- immutable revision 或 split provenance 无法复现；
- C1 route reversal 只在预测器中存在、native continuation 不获益；
- typed delta 在 matched-size baseline 下无 fidelity 优势；
- 任一声明范围内出现 duplicate committed effect；
- closest work 显示 joint formulation 已被完整覆盖。

Concerns unlikely to change before submission:

- 单张 4090D 不是物理 edge-cloud 集群；
- exactly-once 对非 queryable irreversible API 的不可能性；
- 三个 benchmark 不能建立普适 agent migration 结论。

## 16. Action Plan And CCFA Handoffs

Priority: P0  
Action: 修复 policy、validation abort、effect transition 和 stable key 语义。  
Owner skill: ccf-pipeline-orchestrator 协调；implementation 执行；ccf-integrity-auditor 复核。  
Input needed: 当前代码与方法规范。  
Expected output: 新单测、统一文码契约、全部 software gates passed。  
Handoff required: yes。

Priority: P0  
Action: 更新 Cordon/PatchBoard/PatchOptic/QKVShare/edge handover 的文献与差异。  
Owner skill: ccf-literature-searcher。  
Input needed: 本轮 primary-source links。  
Expected output: 更新 BibTeX、related work、closest-axis table 和 baseline status。  
Handoff required: yes。

Priority: P0  
Action: 实现 ALFWorld 与 AppWorld adapter、native trajectory runner 和 effect failure injector。  
Owner skill: ccf-pipeline-orchestrator 协调实现。  
Input needed: pinned official repos、local ML dependencies。  
Expected output: public-data E0/E2/E4 dev artifacts。  
Handoff required: yes。

Priority: P1  
Action: 根据 P0 结果执行正式 E1--E4，再写结果。  
Owner skill: ccf-experiment-designer，随后 ccf-paper-writer。  
Input needed: AutoDL 4090D 与 locked assets。  
Expected output: immutable result package 和 evidence-bound manuscript。  
Handoff required: yes。

Priority: P1  
Action: 生成系统图、sequence 图并做双栏表格 QA。  
Owner skill: ccf-visual-composer。  
Input needed: 最终 taxonomy 和真实结果。  
Expected output: publication-grade figures/tables。  
Handoff required: yes。

Priority: P2  
Action: 在 MLSys 2027 官方规则发布后迁移模板和提交检查。  
Owner skill: ccf-submission-checker。  
Input needed: 官方 2027 CFP/style。  
Expected output: compiled PDF 和 compliance report。  
Handoff required: yes。

Checks run:

- TeX section/label/citation inventory；
- 28/28 citation-key resolution；
- brace 317/317 与 begin/end 28/28 静态平衡；
- 13 个 Python software tests 全通过；
- source-code claim alignment review；
- public-safe closest-work search；
- official MLSys page search。

Checks skipped:

- LaTeX compile 和 PDF visual inspection：本机无 latexmk/pdflatex；
- native model inference：ML dependencies 与 locked model assets 不在本机；
- public benchmark episode：adapters 尚未实现；
- MLSys 2027 compliance：官方规则未找到。

Unresolved risks:

- 最新 2026 年工作仍快速增加；
- formal compute budget 与 wall-clock 未确定；
- AppWorld 当前版本与 effect semantics 需锁定后核验；
- Hera/Cordon/PatchBoard 官方代码可用性未知。
