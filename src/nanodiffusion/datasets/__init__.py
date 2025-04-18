import abc
import torch
import typing as ty

from torch.utils.data._utils.collate import collate # type: ignore
from torch.utils.data import Dataset

from dataclasses import dataclass

from nanoconfig import config
from nanoconfig.experiment import NestedResult

@dataclass
class Sample:
    cond: torch.Tensor | None
    sample: torch.Tensor

    # None if no conditioning or conditioning
    # is meant to be used directly
    num_classes: int | None = None

    def to(self, device: torch.device):
        cond = self.cond.to(device) if self.cond is not None else None
        sample = self.sample.to(device)
        return Sample(cond, sample, self.num_classes)

class SampleDataset(Dataset, ty.Sized, abc.ABC):
    @abc.abstractmethod
    def visualize_batch(self, samples: Sample) -> NestedResult:
        pass

# When we collate a batch of samples, we need to
# keep the num_classes the same
def sample_collate_fn(batch, *, collate_fn_map):
    return Sample(
        collate([b.cond for b in batch], collate_fn_map=collate_fn_map), # type: ignore
        collate([b.sample for b in batch], collate_fn_map=collate_fn_map), # type: ignore
        batch[0].num_classes
    )

import torch.utils.data._utils.collate
torch.utils.data._utils.collate.default_collate_fn_map.update({ # type: ignore
    Sample: sample_collate_fn
})

@config
class DataConfig(abc.ABC):
    @abc.abstractmethod
    def create(self) -> tuple[SampleDataset, SampleDataset]: ...
