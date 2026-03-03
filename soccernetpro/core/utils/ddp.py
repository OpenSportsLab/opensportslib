import torch.distributed as dist
def ddp_setup(rank, world_size):
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"  # any free port
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

def ddp_cleanup():
    dist.destroy_process_group()


import torch
from torch.utils.data import Sampler
import math

class DistributedWeightedSampler(Sampler):
    def __init__(
        self,
        weights,
        num_replicas=None,
        rank=None,
        replacement=True,
        num_samples=None,
        seed=0
    ):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed

        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.dataset_size = len(self.weights)

        # Split dataset across ranks
        self.num_samples_per_rank = math.ceil(self.dataset_size / self.num_replicas)
        self.total_size = self.num_samples_per_rank * self.num_replicas

        # Pad if needed
        if self.total_size > self.dataset_size:
            padding = self.total_size - self.dataset_size
            self.weights = torch.cat([self.weights, self.weights[:padding]])

        # Indices for this rank
        self.rank_indices = list(range(self.rank, self.total_size, self.num_replicas))
        self.rank_weights = self.weights[self.rank_indices]

        # Number of samples to draw
        if num_samples is None:
            self.num_samples = len(self.rank_indices)
        else:
            self.num_samples = num_samples // self.num_replicas

        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)

        sampled = torch.multinomial(
            self.rank_weights,
            self.num_samples,
            self.replacement,
            generator=g
        )

        return (self.rank_indices[i] % self.dataset_size for i in sampled)

    def __len__(self):
        return self.num_samples
