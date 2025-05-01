import abc
import pyarrow as pa
import functools
import pyarrow.dataset as ds
import itertools
import torch.utils.data
import torch.utils._pytree as pytree
import typing as ty
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json
import io
import PIL.Image

import torchvision.transforms.functional

from . import utils

from nanoconfig.data import utils as data_utils
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
    def to_result(self) -> NestedResult: ...

    @abc.abstractmethod
    def with_values(self, sample: dict[str, torch.Tensor] | torch.Tensor) -> ty.Self: ...

@dataclass
class Planar(DataPoint):
    cond_: torch.Tensor | None
    value_: torch.Tensor

    @property
    def cond(self) -> CondValue | None:
        return self.cond_

    @property
    def sample(self) -> SampleValue:
        return self.value_

    def to_result(self) -> NestedResult:
        cond = self.cond_.cpu().numpy() if self.cond_ is not None else None
        value = self.value_.cpu().numpy()
        if cond is not None: value = np.concatenate((cond,value), axis=-1) # type: ignore
        x, y, z = None, None, None
        if value.shape[-1] == 1: x, = value.T
        elif value.shape[-1] == 2: x, y = value.T
        elif value.shape[-1] == 3: x, y, z = value.T
        else: raise ValueError(f"Invalid shape {value.shape}")
        if x is not None and y is not None or False:
            return Figure(go.Figure([go.Scatter(x=list(x), y=list(y), mode="markers",
                    marker_color=z if z is not None else None)]))
        else:
            return Figure(go.Figure([go.Histogram(x=list(x))]))

    def to_dataframe(self) -> pd.DataFrame:
        cond = self.cond_.cpu().numpy() if self.cond_ is not None else None
        value = self.value_.cpu().numpy()
        if cond is not None: value = np.concatenate((cond,value), axis=-1) # type: ignore
        return pd.DataFrame(zip(value.T, ["x", "y", "z"]))

    def with_values(self, sample: SampleValue) -> "Planar":
        assert isinstance(sample, torch.Tensor)
        return Planar(self.cond_, sample)

    @staticmethod
    def from_dataset(dataset: ds.Dataset) -> "ty.Iterator[Planar]":
        mime_type = dataset.schema.metadata.get(b"mime_type", "").decode()
        conditional = mime_type == b"parquet/conditional_planar"
        has_y = "y" in dataset.schema.names
        has_z = "z" in dataset.schema.names

        for batches in dataset.to_batches():
            tensors = []
            tensors.append(torch.tensor(data_utils.as_numpy(batches["x"]).copy()))
            if has_y: tensors.append(torch.tensor(data_utils.as_numpy(batches["y"]).copy()))
            if has_z: tensors.append(torch.tensor(data_utils.as_numpy(batches["z"]).copy()))
            if conditional: cond = tensors.pop(0)
            else: cond = None
            value = torch.stack(tensors, dim=-1)
            yield Planar(cond, value)

pytree.register_pytree_node(
    Planar,
    lambda planar: ([planar.cond_, planar.value_], None),
    lambda children, _: Planar(children[0], children[1]) # type: ignore
)

@dataclass
class Trajectory(DataPoint):
    start: torch.Tensor
    end: torch.Tensor
    points: torch.Tensor
    maze: torch.Tensor

    @property
    def cond(self) -> CondValue | None:
        return {"start": self.start, "end": self.end}

    @property
    def sample(self) -> dict[str, torch.Tensor] | torch.Tensor:
        return self.points

    def to_result(self) -> NestedResult:
        if self.maze.ndim == 4: maze = self.maze[0]
        else: maze = self.maze
        maze = Maze.from_numpy_onehot(maze.cpu().numpy())
        x, y = self.points[:].cpu().numpy().T
        def trajectory(traj):
            x, y = traj.cpu().numpy().T
            return [go.Scatter(x=[x[0]], y=[y[0]*-1], mode="markers", marker_color="green"),
                go.Scatter(x=[x[-1]], y=[y[-1]*-1], mode="markers", marker_color="red"),
                go.Scatter(x=list(x), y=list(y*-1), mode="lines", marker_color="blue")]
        return Figure(go.Figure([
            maze.render_plotly((2, 2), (0,0)),
        ] + list(itertools.chain.from_iterable(trajectory(traj) for traj in self.points)),
        dict(showlegend=False, xaxis_range=[-1.1, 1.1], yaxis_range=[-1.1, 1.1])))

    def with_values(self, sample: SampleValue) -> "Trajectory":
        assert isinstance(sample, torch.Tensor)
        return Trajectory(self.start, self.end, sample, self.maze) # type: ignore

    @staticmethod
    def from_dataset(dataset: ds.Dataset) -> "ty.Iterator[Trajectory]":
        for batch in dataset.to_batches():
            start = torch.tensor(data_utils.as_numpy(batch["start"]).copy())
            end = torch.tensor(data_utils.as_numpy(batch["end"]).copy())
            points = torch.tensor(data_utils.as_numpy(batch["trajectory"]).copy())
            maze = torch.tensor(data_utils.as_numpy(batch["maze"]).copy())
            yield Trajectory(start, end, points, maze)

pytree.register_pytree_node(
    Trajectory, lambda traj: ([traj.start, traj.end, traj.points, traj.maze], None),
    lambda children, aux: Trajectory(children[0], children[1], children[2], children[3]) # type: ignore
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

    def to_result(self) -> NestedResult:
        return ImageResult(self.image)

    def with_values(self, sample: SampleValue) -> "Image":
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
    adapter.register_type("parquet/planar", Planar.from_dataset)
    adapter.register_type("parquet/conditional_planar", Planar.from_dataset)

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
        return pd.DataFrame(rows)
