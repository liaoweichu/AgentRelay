import os
os.environ.setdefault("ALFWORLD_DATA", "/root/autodl-tmp/AgentRelay/datasets/alfworld")
import torch
print("torch", torch.__version__, torch.version.cuda, "cuda_avail", torch.cuda.is_available())
import alfworld, textworld
print("alfworld import OK")
d = os.environ["ALFWORLD_DATA"]
print("ALFWORLD_DATA", d)
top = sorted(os.listdir(d))
print("data top-level:", top[:12])
# check tw-pddl structure
for sub in ["/json_2.1.3_tw-pddl", "/json_2.1.1_json"]:
    p = d + sub
    if os.path.isdir(p):
        print(sub, "exists, train dirs:", sorted(os.listdir(p))[:6])
# try to build a single textworld game from the data
try:
    from alfworld.agents.detector import OracleDetector
    from alfworld.env import TextWorldEnv
    print("TextWorldEnv import OK")
except Exception as e:
    print("env import skipped:", type(e).__name__, e)
print("ALFWORLD_VERIFY_DONE")