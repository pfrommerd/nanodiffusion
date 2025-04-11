from nanoconfig import config

from torch.utils.data import Dataset

import abc

@config
class DataConfig(abc.ABC):
    @abc.abstractmethod
    def create(self) -> tuple[Dataset, Dataset]: ...