import os
os.environ.setdefault("ALFWORLD_DATA", "/root/autodl-tmp/AgentRelay/datasets/alfworld")
import yaml
from alfworld.agents.environment import get_environment

cfg_path = "/root/autodl-tmp/AgentRelay/repositories/alfworld/configs/base_config.yaml"
config = yaml.safe_load(open(cfg_path, encoding="utf-8"))
env_type = config["env"]["type"]
print("env.type =", env_type)
for split in ["train", "eval_in_distribution", "eval_out_of_distribution"]:
    try:
        raw = get_environment(env_type)(config, train_eval=split)
        n = len(raw.game_files)
        print(f"  split={split}: {n} game files; first={os.path.basename(raw.game_files[0]) if n else '-'}")
    except Exception as e:
        print(f"  split={split}: ERROR {type(e).__name__}: {e}")
print("ALFWORLD_ENV_VERIFY_DONE")