#!/usr/bin/env python3
"""Record a REAL network throughput trace on the instance by streaming a public
download and sampling received bytes per second. No synthesis or shuffling."""
import csv
import json
import statistics
import sys
import time
import urllib.request


def _med(values):
    return statistics.median(values)


url = sys.argv[1]
out = sys.argv[2]
duration_s = int(sys.argv[3]) if len(sys.argv) > 3 else 90
chunk = 128 * 1024

samples = []  # (unix_ts, mbps)
start = time.time()
window_start = time.time()
window_bytes = 0
req = urllib.request.Request(url, headers={"User-Agent": "AgentRelay-trace/1.0"})
with urllib.request.urlopen(req, timeout=30) as resp:
    while time.time() - start < duration_s:
        data = resp.read(chunk)
        if not data:
            break
        window_bytes += len(data)
        now = time.time()
        if now - window_start >= 1.0:
            mbps = (window_bytes * 8) / (now - window_start) / 1e6
            samples.append((int(now), round(mbps, 3)))
            window_bytes = 0
            window_start = now

if len(samples) < 30:
    raise SystemExit(f"too few samples captured: {len(samples)}")

with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["unix_ts", "mbps"])
    for ts, mbps in samples:
        w.writerow([ts, mbps])

rates = [m for _, m in samples]
print(f"samples={len(samples)} min={min(rates):.1f} med={_med(rates):.1f} max={max(rates):.1f} mbps")
print(f"wrote {out}")