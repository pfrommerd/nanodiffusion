import abc
import pyarrow as pa
import pyarrow.dataset as ds
import torch.utils.data
import torch.utils._pytree as pytree
import typing as ty
import pandas as pd
import json
import io
import PIL.Image

import torchvision.transforms.functional

from . import utils

from nanoconfig.data.torch import TorchAdapter
from nanoconfig.data.visualizer import DataVisualizer
from nanoconfig.experiment import NestedResult, Figure, Image as ImageResult
from dataclasses import dataclass

from mazelib.maze import Maze

@dataclass(frozen=True)
class DiscreteLabel:
    idx: int
    num_classes: int

CondValue = dict[str, torch.Tensor | DiscreteLabel] | torch.Tensor | DiscreteLabel
SampleValue = dict[str, torch.Tensor] | torch.Tensor

class DataPoint(abc.ABC):
    def to(self, device: torch.device) -> ty.Self:
        return pytree.tree_map(
            lambda x: x.to(device) if isinstance(x, torch.Tensor) else x,
            self
        )

    @property
    def has_cond(self) -> bool:
        return self.cond is not None

    @property
    @abc.abstractmethod
    def cond(self) -> CondValue | None:
        pass

    @property
    @abc.abstractmethod
    def sample(self) -> dict[str, torch.Tensor] | torch.Tensor:
        pass

    @abc.abstractmethod
    def visualize(self) -> NestedResult: ...

    @staticmethod
    @abc.abstractmethod
    def from_values(cond: dict[str, torch.Tensor | DiscreteLabel] | torch.Tensor | DiscreteLabel,
        sample: dict[str, torch.Tensor]) -> "DataPoint": ...

@dataclass
class Trajectory(DataPoint):
    start: torch.Tensor
    end: torch.Tensor
    points: torch.Tensor

    @property
    def cond(self) -> CondValue | None:
        return {"start": self.start, "end": self.end}

    @property
    def sample(self) -> dict[str, torch.Tensor] | torch.Tensor:
        return self.points

    def visualize(self) -> NestedResult:
        fig = None
        return Figure(fig)

    @ty.override
    def to(self, device: torch.device) -> "Trajectory":
        return Trajectory(self.start.to(device), self.end.to(device), self.points.to(device))

    @staticmethod
    def from_values(cond: CondValue | None, sample: SampleValue) -> "Trajectory":
        return Trajectory(cond["start"], cond["end"], sample["points"]) # type: ignore

    @staticmethod
    def from_dataset(dataset: ds.Dataset) -> "ty.Iterator[Trajectory]":
        T = dataset.schema.field("trajectory").type.list_size
        for batch in dataset.to_batches():
            start = batch["start"].flatten().to_numpy(zero_copy_only=False)
            end = batch["end"].flatten().to_numpy(zero_copy_only=False)
            points = batch["trajectory"].flatten().flatten().to_numpy(zero_copy_only=False)
            start = torch.from_numpy(start.reshape(-1, 2))
            end = torch.from_numpy(end.reshape(-1, 2))
            points = torch.from_numpy(points.reshape(-1, T, 2))
            yield Trajectory(start, end, points)

pytree.register_pytree_node(
    Trajectory, lambda traj: ([traj.start, traj.end, traj.points], None),
    lambda children, aux: Trajectory(children[0], children[1], children[2]) # type: ignore
)

@dataclass
class Image(DataPoint):
    image: torch.Tensor

    @property
    def cond(self) -> CondValue | None:
        return None

    @property
    def sample(self) -> SampleValue:
        return self.image

    def to(self, device: torch.device) -> "Image":
        return Image(self.image.to(device))

    def visualize(self) -> NestedResult:
        return ImageResult(self.image)

    @staticmethod
    def from_values(cond: CondValue | None, sample: SampleValue) -> "Image":
        assert isinstance(sample, torch.Tensor)
        return Image(sample)

    @staticmethod
    def from_dataset(dataset: ds.Dataset) -> "ty.Iterator[Image]":
        for batch in dataset.to_batches():
            imgs = []
            for row in batch.to_pylist():
                image = row["image"]["bytes"]
                image = PIL.Image.open(io.BytesIO(image))
                image = torchvision.transforms.functional.to_tensor(image)
                imgs.append(Image(image))
            yield pytree.tree_map(lambda *x: torch.stack(x, dim=0), *imgs)

pytree.register_pytree_node(
    Image, lambda image: ([image.image], None),
    lambda children, aux: Image(children[0]) # type: ignore
)

@dataclass
class LabeledImage(Image):
    label: torch.Tensor
    classes: list[str]

    @ty.override
    def visualize(self) -> NestedResult: # type: ignore
        fig = None
        return Figure(fig)

    @staticmethod
    def from_dataset(dataset: ds.Dataset) -> "ty.Iterator[LabeledImage]":
        classes = json.loads(dataset.schema.metadata.get(b"classes", "[]"))
        for batch in dataset.to_batches():
            imgs = []
            for row in batch.to_pylist():
                image = row["image"]["bytes"]
                label = row["label"]
                image = PIL.Image.open(io.BytesIO(image.as_py()))
                image = torchvision.transforms.functional.to_tensor(image)
                imgs.append(LabeledImage(image, torch.tensor(label), classes))
            yield pytree.tree_map(lambda *x: torch.stack(x, dim=0), *imgs)

pytree.register_pytree_node(
    LabeledImage, lambda image: ([image.image, image.label], image.classes),
    lambda children, aux: LabeledImage(children[0], children[1], aux.classes) # type: ignore
)

def register_types(adapter: TorchAdapter[DataPoint]):
    adapter.register_type("parquet/trajectory", Trajectory.from_dataset)
    adapter.register_type("parquet/image", Image.from_dataset)
    adapter.register_type("parquet/image+label", LabeledImage.from_dataset)

class Visualizer(DataVisualizer):
    def as_dataframe(self, split):
        adapter = TorchAdapter[DataPoint]()
        register_types(adapter)
        rows = []
        for batch in adapter.convert(split):
            N = utils.axis_size(batch, 0)
            for i in range(N):
                r = pytree.tree_map(lambda x: x[i], batch).visualize()
                rows.append(r if isinstance(r, dict) else {"data": r})
                if len(rows) > 200:
                    break
        return pd.DataFrame(rows)
