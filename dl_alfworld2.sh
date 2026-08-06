#!/usr/bin/env bash
set -u
DATA_DIR="/root/autodl-tmp/AgentRelay/datasets/alfworld"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

URL="https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip"
MIRRORS="https://gh-proxy.com https://mirror.ghproxy.com https://ghfast.top https://ghproxy.net"

echo "== test mirrors =="
WORKING=""
for m in $MIRRORS; do
  code=$(timeout 12 curl -sI -o /dev/null -w '%{http_code}' "$m/$URL" 2>/dev/null)
  echo "  $m => $code"
  if [ "$code" = "200" ] || [ "$code" = "302" ]; then
    WORKING="$m"
    break
  fi
done

if [ -z "$WORKING" ]; then
  echo "NO_WORKING_MIRROR"
  exit 1
fi
echo "USING_MIRROR=$WORKING"

echo "== download 3 zips =="
for spec in "0.2.2/json_2.1.1_json.zip" "0.2.2/json_2.1.1_pddl.zip" "0.4.2/json_2.1.3_tw-pddl.zip"; do
  name=$(basename "$spec")
  echo "-- $name --"
  curl -L --connect-timeout 15 --max-time 600 --retry 5 --retry-delay 3 -C - -o "$name" "$WORKING/https://github.com/alfworld/alfworld/releases/download/$spec"
  echo "   size: $(stat -c%s "$name" 2>/dev/null) bytes"
done

echo "== extract =="
for z in json_2.1.1_json.zip json_2.1.1_pddl.zip json_2.1.3_tw-pddl.zip; do
  echo "   unzip $z"
  unzip -oq "$z" || echo "   WARN unzip failed"
  rm -f "$z"
done

echo "== logic files =="
mkdir -p "$DATA_DIR/logic"
REPO="/root/autodl-tmp/AgentRelay/repositories/alfworld"
cp -f "$REPO/alfworld/data/alfred.pddl" "$DATA_DIR/logic/alfred.pddl" 2>/dev/null || echo "   (no pddl)"
cp -f "$REPO/alfworld/data/alfred.twl2" "$DATA_DIR/logic/alfred.twl2" 2>/dev/null || echo "   (no twl2)"

echo "== result =="
du -sh "$DATA_DIR"
echo "ALF_DL_DONE"