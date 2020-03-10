import os
import sys
import torch
import random
import numpy as np

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Optional


def load_library(conf, return_policy_network: bool = False):
    lib = []
    pathli = conf.experts_path
    if isinstance(conf.experts_path, str):
        pathli = [conf.experts_path]
    for path in pathli:
        with add_path(path):
            mod = __import__('nn').setup
            print(mod)
            policy = mod(conf.env.observation_nd, conf.env.action_nd)
        policy.to(conf.device)

        @torch.no_grad()
        def act(obses):
            return policy(obses)
        lib.append(act)
    if return_policy_network:
        assert len(lib) == 1
        return policy
    return lib


class BaseMemory(ABC):
    @abstractmethod
    def add(self):
        pass

    @abstractmethod
    def sample(self):
        pass


class ReplayBuffer(BaseMemory):
    def __init__(self, size):
        """Create Replay buffer.
        Parameters
        ----------
        size: int
            Max number of transitions to store in the buffer. When the buffer
            overflows the old memories are dropped.
        """
        self._storage = []
        self._maxsize = size
        self._next_idx = 0

    def __len__(self):
        return len(self._storage)

    def add(self, *args):
        if self._next_idx >= len(self._storage):
            self._storage.append(args)
        else:
            self._storage[self._next_idx] = args
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, idxes):
        res = defaultdict(list)
        for i in idxes:
            data = self._storage[i]
            for k, v in enumerate(data):
                res[k].append(v)
        return [np.array(v) for v in res.values()]

    def sample(self, batch_size, *args, **kwargs):
        """Sample a batch of experiences."""
        idxes = [random.randint(0, len(self._storage) - 1) for _ in range(batch_size)]
        return self._encode_sample(idxes), np.ones(batch_size), idxes



class add_path():
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        sys.path.insert(0, self.path)

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            sys.path.remove(self.path)
        except ValueError:
            pass
