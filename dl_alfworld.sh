#!/usr/bin/env bash
set -e
DATA_DIR="/root/autodl-tmp/AgentRelay/datasets/alfworld"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

dl() {
  url="$1"; name="$2"; dest="$3"
  echo "== $name =="
  curl -L --retry 8 --retry-delay 3 -C - -o "$name" "$url"
  echo "   downloaded: $(stat -c%s "$name") bytes"
}

dl "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_json.zip" json_2.1.1_json.zip "$DATA_DIR"
dl "https://github.com/alfworld/alfworld/releases/download/0.2.2/json_2.1.1_pddl.zip" json_2.1.1_pddl.zip "$DATA_DIR"
dl "https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip" json_2.1.3_tw-pddl.zip "$DATA_DIR"

echo "== extracting =="
for z in json_2.1.1_json.zip json_2.1.1_pddl.zip json_2.1.3_tw-pddl.zip; do
  echo "   unzip $z"
  unzip -oq "$z" || echo "   WARN unzip failed for $z"
  rm -f "$z"
done

echo "== logic files =="
mkdir -p "$DATA_DIR/logic"
REPO="/root/autodl-tmp/AgentRelay/repositories/alfworld"
cp -f "$REPO/alfworld/data/alfred.pddl" "$DATA_DIR/logic/alfred.pddl" 2>/dev/null || echo "   (no alfred.pddl at expected path)"
cp -f "$REPO/alfworld/data/alfred.twl2" "$DATA_DIR/logic/alfred.twl2" 2>/dev/null || echo "   (no alfred.twl2 at expected path)"

echo "== result =="
du -sh "$DATA_DIR"
ls "$DATA_DIR"
echo "ALF_DL_DONE"