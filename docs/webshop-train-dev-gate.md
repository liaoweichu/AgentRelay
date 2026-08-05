# WebShop train/dev router gate

This gate never reads the official WebShop test range. The upstream seed-233
goal order is fixed as test `0..499`, dev `500..1499`, and train `1500..12086`.
The diagnostic samples below are drawn only inside train and dev.

Run on the AutoDL 4090D from `/root/autodl-tmp/AgentRelay` after pulling the
current commit. Use the locked Gemma 4 configuration and measured service
profile already present on the instance.

```bash
REVISION=64fa2a5c15c7daa698b9ac93f5bb5437b634c9bd
WEBSHOP_FILE=/root/autodl-tmp/AgentRelay/repositories/webshop/data/items_shuffle.json
CONFIG=/root/autodl-tmp/AgentRelay/formal-autodl-4090d.locked.json
PROFILE=/root/autodl-tmp/AgentRelay/results/service-profile.json

python scripts/build_official_task_manifest.py \
  --benchmark webshop --split train --purpose train \
  --revision "$REVISION" --webshop-file-path "$WEBSHOP_FILE" \
  --sample-count 200 --sample-seed 20260805 \
  --output results/manifests/webshop-train-200.json

python scripts/build_official_task_manifest.py \
  --benchmark webshop --split dev --purpose tune \
  --revision "$REVISION" --webshop-file-path "$WEBSHOP_FILE" \
  --sample-count 100 --sample-seed 20260806 \
  --output results/manifests/webshop-dev-100.json
```

Run each fixed endpoint in a separate process. This keeps only one Gemma model
resident on the 24 GB GPU at a time.

```bash
python scripts/run_autodl_matrix.py "$CONFIG" \
  results/manifests/webshop-train-200.json "$PROFILE" --method edge_only
python scripts/run_autodl_matrix.py "$CONFIG" \
  results/manifests/webshop-train-200.json "$PROFILE" --method cloud_only
python scripts/run_autodl_matrix.py "$CONFIG" \
  results/manifests/webshop-dev-100.json "$PROFILE" --method edge_only
python scripts/run_autodl_matrix.py "$CONFIG" \
  results/manifests/webshop-dev-100.json "$PROFILE" --method cloud_only
```

Record the four printed `run_directory` values, then combine the two train runs
and the two dev runs:

```bash
python scripts/collect_endpoint_episodes.py \
  TRAIN_EDGE_RUN TRAIN_CLOUD_RUN --split train \
  --output results/webshop-train-200-paired.jsonl

python scripts/collect_endpoint_episodes.py \
  DEV_EDGE_RUN DEV_CLOUD_RUN --split dev \
  --output results/webshop-dev-100-paired.jsonl

python scripts/run_router_learnability_gate.py \
  results/webshop-train-200-paired.jsonl \
  results/webshop-dev-100-paired.jsonl \
  results/router-webshop-task-reward.joblib \
  results/router-webshop-train-dev-gate.json
```

The gate passes only when all predeclared checks hold: 100 paired dev tasks,
positive oracle gap, router reward no lower than the best fixed endpoint, at
least 30% oracle-gap capture, and cloud selection between 10% and 90%. A failed
gate exits with status 2 and must not trigger an official-test run.
