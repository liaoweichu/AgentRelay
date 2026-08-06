# AutoDL 4090D 交接：Gemma 4 × τ²-bench

日期：2026-08-06
主实验模型：edge `google/gemma-4-E4B-it`，cloud `google/gemma-4-12b-it`
模型来源：ModelScope
主工作负载：τ²-bench airline / retail / telecom text mode
本阶段目标：先完成三域 smoke、E4B 精度敏感性和 train/dev reward-aware router 可学习性门控；不运行 τ² 官方 test，不写论文。

## 冻结约束

- τ²-bench 固定到 `0ed2fd8d830a20657d89ae9c2efcc94838aa7129`。不得使用已演化为 τ³ 的当前 `main`。
- router 的内部 train/dev 均只从 τ² 官方 `train` 划分；官方 `test` 保持封存。
- 正式成对端点统一使用 NF4 4-bit 权重、BF16 compute。E4B FP16 只出现在独立 sensitivity 目录，不能混入 router 训练。
- user simulator 固定为 τ² 官方实现，模型与解码见 `configs/tau2-user-simulator.json`。密钥只从 `OPENAI_API_KEY` 读取，不写入任何配置或结果。
- edge 与 cloud 串行、单模型驻留。任何 runner 重启都使用同一命令和同一输出目录，由内置 provenance 校验决定是否可恢复。
- 无需人工记录 Git 哈希或上传文件哈希；代码内部仍自动保存不可变 revision 和 artifact self-hash。

## 执行表

| 顺序 | 动作 | 通过条件 | 失败时处理 |
| --- | --- | --- | --- |
| G0 | 拉取最新 `main` | 工作树干净，fast-forward 成功 | 有云端本地改动或冲突就停止；不要 reset |
| G1 | 安装项目固定依赖 | `modelscope==1.39.1`，测试与编译通过 | 返回完整安装/测试日志 |
| G2 | 生成不可变配置 | 两个 ModelScope 模型均解析为完整 revision；τ² 为指定 commit | 不允许回退到 HF 或 `master` 快照 |
| G3 | checkout 并安装 τ² | origin、HEAD、三域 split 全部通过 | commit/split 不符立即停止 |
| G4 | GPU/ModelScope/τ² 预检 | 4090D、显存、缓存路径、包版本、模型对、user secret 全部通过 | 模型尚未加载前停止 |
| G5 | 生成 τ² 内部 train/dev manifest | airline 21/9、retail 52/22、telecom 52/22 | 任何 test 访问或计数变化均停止 |
| G6 | 生成新服务 profile | 精确记录 ID/revision/source/dtype/quantization，真实网络 trace 非空 | 不复用旧 Qwen profile，不填常数带宽 |
| G7 | 两模型三域 text smoke | 三个 domain 均有一对 episode；step-zero features 成对一致 | parser、API、OOM 或输入不一致即停止 |
| G8 | E4B 4-bit/FP16 sensitivity | 两臂任务完全一致，分别记录 reward/latency/peak memory | 仅返回诊断；不得据此改写正式配置 |
| G9 | τ² train/dev reward router gate | 125 train tasks、53 dev tasks 均有 edge/cloud 配对；输出 router 和 gate | exit 2 是经验门控失败；其他非零是运行/完整性失败 |
| G10 | 回传审计包 | 下列文件和完整日志齐全 | 在本地审计前不跑 test/baseline/full matrix |

## 0. 更新代码

```bash
cd /root/autodl-tmp/AgentRelay
git status --short
git fetch origin
git checkout main
git pull --ff-only origin main
git diff --exit-code
git diff --cached --exit-code
```

首个 `git status --short` 非空时停止并报告。不要覆盖云端本地文件。

## 1. 安装与软件门控

```bash
cd /root/autodl-tmp/AgentRelay
bash scripts/bootstrap_autodl.sh "$PWD"
source /root/autodl-tmp/AgentRelay/env.sh

python -m pip show modelscope
python -c 'import modelscope; assert modelscope.__version__=="1.39.1"; print(modelscope.__version__)'
python -m pytest -q
python -m compileall -q src scripts tests
python scripts/audit_model_pair.py
```

## 2. 生成锁定配置

```bash
ROOT=/root/autodl-tmp/AgentRelay
LOCK=$ROOT/formal-autodl-4090d.locked.json

python scripts/lock_config.py \
  configs/formal-autodl-4090d.template.json \
  "$LOCK"
python -m agentrelay.cli check-config "$LOCK"
```

锁定过程直接访问 ModelScope API。两个模型的 `model_source` 必须仍为 `modelscope`；不得改为 Hugging Face ID，也不得把本地目录名伪装成 revision。

## 3. checkout、安装 τ²、准备公共数据

```bash
python scripts/download_public_data.py "$LOCK"
python scripts/checkout_repositories.py "$LOCK"

TAU2_REPO=$ROOT/repositories/tau2-bench
bash scripts/prepare_tau2_bench.sh "$TAU2_REPO"

git -C "$TAU2_REPO" rev-parse HEAD
git -C "$TAU2_REPO" remote get-url origin
```

预期 HEAD：

```text
0ed2fd8d830a20657d89ae9c2efcc94838aa7129
```

## 4. 云端预检

先在 shell 中提供 user simulator 密钥；不要写入文件：

```bash
export OPENAI_API_KEY='在云端 shell 中设置，不要回传'

python scripts/preflight.py \
  "$LOCK" \
  --output "$ROOT/results/preflight-gemma4.json"

python scripts/preflight_tau2.py \
  "$LOCK" \
  "$TAU2_REPO" \
  "$ROOT/configs/tau2-user-simulator.json" \
  --output "$ROOT/results/preflight-tau2.json"
```

`preflight-tau2.json` 应报告官方 split：airline 30/20、retail 74/40、telecom 74/40（train/test）。这里只读取 split ledger，router 不读取 test task 内容或 label。

## 5. 固定内部 train/dev 清单

```bash
MANIFEST=$ROOT/results/tau2-router-splits.json

python scripts/build_tau2_task_manifest.py \
  "$TAU2_REPO" \
  "$MANIFEST" \
  --split-seed 20260806 \
  --dev-fraction 0.30
```

预期内部计数：

```text
airline train=21 dev=9
retail  train=52 dev=22
telecom train=52 dev=22
total   train=125 dev=53
```

## 6. 重新生成 Gemma 服务 profile

沿用之前真实采集的网络 trace；旧 service profile 只能用于找回 trace 的来源说明，不能作为本轮模型延迟。

```bash
TRACE=$ROOT/datasets/network/trace.csv
OLD_PROFILE=$ROOT/results/service-profile.json
PROFILE=$ROOT/results/service-profile-gemma4.json

test -s "$TRACE"
test -s "$OLD_PROFILE"
TRACE_SOURCE="$(python -c 'import json; print(json.load(open("/root/autodl-tmp/AgentRelay/results/service-profile.json"))["network_trace"]["source"])')"
test -n "$TRACE_SOURCE"

python scripts/profile_models.py \
  "$LOCK" \
  "$PROFILE" \
  --repeats 3 \
  --network-trace "$TRACE" \
  --rate-column mbps \
  --sample-period-ms 1000 \
  --trace-source "$TRACE_SOURCE"
```

profile 中每个 role 必须同时包含 `model_id`、`model_revision`、`model_source`、`dtype` 和 `quantization`。本命令先 E4B 后 12B，顺序加载并释放。

## 7. 三域 text-mode smoke

```bash
SMOKE=$ROOT/results/tau2-text-smoke

python scripts/run_tau2_text_smoke.py \
  "$LOCK" \
  "$PROFILE" \
  "$MANIFEST" \
  "$TAU2_REPO" \
  "$ROOT/configs/tau2-user-simulator.json" \
  "$SMOKE"
```

必须生成 `smoke-report.json`，且 `domains` 为 airline/retail/telecom、`paired_tasks=3`。此步通过后才进入 sensitivity。

## 8. E4B FP16/4-bit sensitivity

```bash
SENS=$ROOT/results/tau2-e4b-precision-sensitivity

python scripts/run_tau2_precision_sensitivity.py \
  "$LOCK" \
  "$PROFILE" \
  "$MANIFEST" \
  "$TAU2_REPO" \
  "$ROOT/configs/tau2-user-simulator.json" \
  "$SENS" \
  --tasks-per-domain 3
```

此命令串行运行 E4B 正式 NF4/BF16-compute 和 E4B FP16，共 9 个相同任务。结果只用于判断量化敏感性；无论结果如何，下一步正式成对 gate 仍由锁定配置控制为 E4B/12B 同为 4-bit。

## 9. train/dev reward-aware router 门控

```bash
GATE=$ROOT/results/tau2-train-dev-gate

python scripts/run_tau2_train_dev_gate.py \
  "$LOCK" \
  "$PROFILE" \
  "$MANIFEST" \
  "$TAU2_REPO" \
  "$ROOT/configs/tau2-user-simulator.json" \
  "$GATE"
STATUS=$?
echo "tau2_gate_exit=$STATUS"
```

该命令严格串行运行：train edge → train cloud → dev edge → dev cloud；每个任务单独原子落盘。SSH 中断后，用完全相同的命令和输出目录重跑即可。

- exit 0：预声明可学习性门控通过。
- exit 2：实验完整执行，但 router 门控未达标。回传所有文件并停止。
- 其他非零：依赖、API、模型、provenance 或任务执行错误。不要解释为经验失败。

若固定 user simulator 的 step-zero 输出在 edge/cloud 两次执行中不一致，配对会 fail closed；不能手工修改 features 或 episode 让其通过。

## 10. 回传文件

回传以下内容和完整 console log：

1. `formal-autodl-4090d.locked.json`
2. `results/preflight-gemma4.json`
3. `results/preflight-tau2.json`
4. `results/tau2-router-splits.json`
5. `results/service-profile-gemma4.json`
6. `results/tau2-text-smoke/` 全目录
7. `results/tau2-e4b-precision-sensitivity/` 全目录
8. `results/tau2-train-dev-gate/` 全目录
9. 最终 `tau2_gate_exit`

不要回传 `OPENAI_API_KEY`。不需要额外制作 SHA256 sidecar，也不需要人工抄录 Git 哈希；runner 的 `run-context.json`、episode hash 和 receipt 已自动记录这些约束。

在本地审计这些文件之前，不运行 τ² test、WebShop 扩展、baseline matrix、ablation 或论文写作。
