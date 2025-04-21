import abc
import torch
import typing as ty
import json

from torch.utils.data._utils.collate import collate # type: ignore
from torch.utils.data import Dataset

from dataclasses import dataclass

from nanoconfig import config
from nanoconfig.experiment import NestedResult, Experiment

T = ty.TypeVar('T')

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
    def __init__(self, config, seed, experiment : None | Experiment):
        self.config = config
        self.seed = seed
        self.experiment = experiment
        self.data = None

    def _populate(self):
        self.data = list(self.generate(self.seed))
        # name = type(self).__name__.lower()
        # if self.experiment is not None:
        #     artifact_id = self.experiment.find_artifact(f"dataset_{name}", type="dataset")
        #     if not artifact_id:
        #         self.data = list(self.generate(self.seed))
        #         with self.experiment.create_artifact(f"dataset_{name}", type="dataset") as builder:
        #             artifact = builder.build()
        #     else:
        #         artifact = self.experiment.use_artifact(artifact_id)
        # else:

    def __hash__(self):
        d = json.dumps(self.config, sort_keys=True)
        return hash((type(self), self.seed, d))

    def __getitem__(self, index) -> Sample:
        if self.data is None:
            self._populate()
        assert self.data is not None
        return self.data[index]

    def __len__(self) -> int:
        if self.data is None:
            self._populate()
        assert self.data is not None
        return len(self.data)

    # Generate the data
    @abc.abstractmethod
    def generate(self, seed : int) -> ty.Iterator[Sample]:
        pass

    @abc.abstractmethod
    def visualize_batch(self, samples: Sample) -> NestedResult:
        pass

    @property
    def in_memory(self) -> bool:
        return True

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
    def create(self, experiment : Experiment | None = None) -> tuple[SampleDataset, SampleDataset]: ...
