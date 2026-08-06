import json
from pathlib import Path

pilot = Path("/root/autodl-tmp/AgentRelay/results/manifests/webshop-pilot-200.json")
old = Path("/root/autodl-tmp/AgentRelay/results/manifests/webshop-test-200.json")

for name, path in (("pilot-200", pilot), ("old-test-200", old)):
    d = json.loads(path.read_text())
    print(name, "tasks=", len(d.get("tasks", [])),
          "complete=", d.get("complete_official_split"),
          "split=", d.get("split"),
          "dataset_revision=", d.get("dataset_revision"))

# check the synced build script on cloud
import subprocess
r = subprocess.run(
    ["grep", "-n", "WEBSHOP_OFFICIAL_SPLIT_SIZES", "/root/autodl-tmp/AgentRelay/scripts/build_official_task_manifest.py"],
    capture_output=True, text=True)
print("cloud script has official split sizes:", r.returncode == 0)