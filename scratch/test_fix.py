import random
import numpy as np
import torch
from transformers import set_seed

# Mock config object
class Config:
    seed_split = 42
    bf16 = False
    fp16 = False

cfg = Config()

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

try:
    set_global_seed(cfg.seed_split)
    print("Successfully set global seed!")
except Exception as e:
    print(f"Error: {e}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")
