#!/usr/bin/env bash
rm -f /root/autodl-tmp/AgentRelay/tmp_gpu_check.sh /root/autodl-tmp/AgentRelay/tmp_preflight.sh
for r in alfworld webshop appworld; do
  echo -n "$r="
  git -C /root/autodl-tmp/AgentRelay/repositories/$r rev-parse HEAD
done