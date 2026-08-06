#!/usr/bin/env bash
echo "---check for stdlib-shadowing dirs---"
for base in /root/autodl-tmp/AgentRelay/repositories/webshop /root/autodl-tmp/AgentRelay/src /root/autodl-tmp/AgentRelay/repositories/appworld/src; do
  echo "== $base =="
  ls -d "$base"/[a-z]* 2>/dev/null | xargs -n1 basename 2>/dev/null
done
echo "---venv site-packages xml---"
ls -d /root/autodl-tmp/AgentRelay/venv/lib/python3.12/site-packages/xml* 2>/dev/null
echo "---/root cwd xml---"
ls -d /root/xml* /root/locale* 2>/dev/null
echo "---proc python cwd---"
for p in $(pgrep -f "vn/bin/python" | head -3); do echo "pid $p cwd:"; readlink /proc/$p/cwd 2>/dev/null; done