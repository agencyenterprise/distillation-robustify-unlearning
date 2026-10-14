import os

# Each path can be overridden by an environment variable of the same name; defaults are upstream's.
CACHE_DIR = os.environ.get('CACHE_DIR', '/workspace/.cache')
DATASET_DIR = os.environ.get('DATASET_DIR', '/workspace/datasets')
MODEL_DIR = os.environ.get('MODEL_DIR', '/workspace/models/non-wmdp')
WMDP_MODEL_DIR = os.environ.get('WMDP_MODEL_DIR', '/workspace/models/wmdp')
WANDB_API_KEY_PATH = "tokens/wandb_token.txt"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Upstream always pointed HF_HOME at CACHE_DIR. Shared clusters set HF_HUB_CACHE themselves,
# so this is now opt-in: SET_HF_HOME=1.
if os.environ.get('SET_HF_HOME') == '1':
    os.environ["HF_HOME"] = CACHE_DIR


def _read_wandb_key():
    if os.environ.get('WANDB_API_KEY'):
        return os.environ['WANDB_API_KEY']
    try:
        with open(WANDB_API_KEY_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return ""


# W&B is opt-in: USE_WANDB=1 plus a key (WANDB_API_KEY or tokens/wandb_token.txt).
# Otherwise it is disabled and the local JSONL records are the only log.
WANDB_API_KEY = _read_wandb_key()
USE_WANDB = os.environ.get('USE_WANDB') == '1' and bool(WANDB_API_KEY)
if not USE_WANDB:
    os.environ["WANDB_MODE"] = "disabled"
