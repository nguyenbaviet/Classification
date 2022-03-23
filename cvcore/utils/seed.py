from datetime import datetime
import numpy as np
import os
import random
import torch


def seed_all(seed=None):
    if seed is None:
        seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
        )
        print("Using a generated random seed {}".format(seed))
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def worker_init_reset_seed(worker_id):
    initial_seed = torch.initial_seed() % 2 ** 31
    seed_all(initial_seed + worker_id)
