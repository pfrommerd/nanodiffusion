import abc
import math
import pyarrow as pa
import functools
import pyarrow.dataset as ds
import hashlib
import itertools
import torch.utils.data
import torch.utils._pytree as pytree
import typing as ty
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import json
import logging
import io
import PIL.Image

import torchvision.transforms.functional

from . import utils

from nanoconfig.data import utils as data_utils
from nanoconfig.data import Data
from nanoconfig.data.source import DataRepository
from nanoconfig.data.transform import DataTransform, DataPipeline
from nanoconfig.data.torch import TorchAdapter
from nanoconfig.data.visualizer import DataVisualizer
from nanoconfig.experiment import (
    NestedResult, Figure as FigureResult, Image as ImageResult,
    Table as TableResult
)
from dataclasses import dataclass

from sklearn.manifold import TSNE

from mazelib.maze import Maze

logger = logging.getLogger(__name__)

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
    def sample(self) -> SampleValue:
        pass

    @abc.abstractmethod
    def to_dataframe(self) -> pd.DataFrame:
        pass

    @abc.abstractmethod
    def to_result(self) -> NestedResult: ...

    @abc.abstractmethod
    def with_values(self, sample: SampleValue, cond: CondValue | None = None) -> ty.Self: ...

@dataclass
class Point(DataPoint):
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
            return FigureResult(go.Figure([go.Scatter(x=list(x), y=list(y), mode="markers",
                    marker_color=z if z is not None else None)]))
        else:
            return FigureResult(go.Figure([go.Histogram(x=list(x))]))

    def to_dataframe(self) -> pd.DataFrame:
        cond = self.cond_.cpu().numpy() if self.cond_ is not None else None
        value = self.value_.cpu().numpy()
        if cond is not None: value = np.concatenate((cond,value), axis=-1) # type: ignore
        return pd.DataFrame(zip(value.T, ["x", "y", "z"]))

    def with_values(self, sample: SampleValue, cond: CondValue | None = None) -> "Point":
        assert isinstance(sample, torch.Tensor)
        return Point(self.cond_ if cond is None else cond, sample) # type: ignore

    @staticmethod
    def from_dataset(dataset: ds.Dataset) -> "ty.Iterator[Point]":
        mime_type = dataset.schema.metadata.get(b"mime_type", "").decode()
        conditional = mime_type == "data/cond_point"
        has_y = "y" in dataset.schema.names
        has_z = "z" in dataset.schema.names

        for batches in dataset.to_batches():
            tensors = []
            tensors.append(torch.tensor(data_utils.as_numpy(batches["x"]).copy()))
            if has_y: tensors.append(torch.tensor(data_utils.as_numpy(batches["y"]).copy()))
            if has_z: tensors.append(torch.tensor(data_utils.as_numpy(batches["z"]).copy()))
            if conditional: cond = tensors.pop(0)[:,None]
            else: cond = None
            value = torch.stack(tensors, dim=-1)
            yield Point(cond, value)

pytree.register_pytree_node(
    Point,
    lambda planar: ([planar.cond_, planar.value_], None),
    lambda children, _: Point(children[0], children[1]) # type: ignore
)

@dataclass
class Trajectory(DataPoint):
    start: torch.Tensor
    end: torch.Tensor
    points: torch.Tensor
    maze: torch.Tensor

    @property
    def cond(self) -> CondValue | None:
        return self.start
        # return {"start": self.start} #

    @property
    def sample(self) -> dict[str, torch.Tensor] | torch.Tensor:
        return self.points

    def to_dataframe(self) -> pd.DataFrame:
        if self.maze.ndim == 4: maze = self.maze[0]
        else: maze = self.maze
        maze = Maze.from_numpy_onehot(maze.cpu().numpy())
        x, y = self.points[:].cpu().numpy().T
        assert False

    def to_result(self) -> NestedResult:
        if self.maze.ndim == 4: maze = self.maze[0]
        else: maze = self.maze
        maze = Maze.from_numpy_onehot(maze.cpu().numpy())
        x, y = self.points[:].cpu().numpy().T
        def trajectory(traj):
            x, y = traj.cpu().numpy().T
            return [go.Scatter(x=[x[0]], y=[y[0]*-1], mode="markers", marker_color="green"),
                go.Scatter(x=[x[-1]], y=[y[-1]*-1], mode="markers", marker_color="red"),
                go.Scatter(x=list(x), y=list(y*-1), mode="lines")]
        return FigureResult(go.Figure([
            maze.render_plotly((2, 2), (0,0)),
        ] + list(itertools.chain.from_iterable(trajectory(traj) for traj in self.points)),
            layout=dict(showlegend=False, xaxis_range=[-1.1, 1.1], yaxis_range=[-1.1, 1.1])))

    def with_values(self, sample: SampleValue, cond: CondValue | None = None) -> "Trajectory":
        assert isinstance(sample, torch.Tensor)
        start = self.start if cond is None else cond["start"] # type: ignore
        # end = self.end if cond is None else cond["end"] # type: ignore
        end = sample[:,-1] if sample.ndim == 3 else sample[-1]
        maze = self.maze if cond is None or "maze" not in cond else cond["maze"] # type: ignore
        return Trajectory(start, end, sample, maze) # type: ignore

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
    # Optionally a label and an embedding
    label: torch.Tensor | None = None
    embed: torch.Tensor | None = None

    @property
    def cond(self) -> CondValue | None:
        if self.embed is not None:
            return self.embed
        return None

    @property
    def sample(self) -> SampleValue:
        return self.image

    def to_dataframe(self):
        if self.image.ndim == 4:
            images = (self.image.cpu().numpy()*255).astype(np.uint8).squeeze(-1)
            columns : dict = {"image": [PIL.Image.fromarray(images[i]) for i in range(images.shape[0])]}
            if self.label is not None:
                columns["label"] = self.label.cpu().numpy()
            if self.embed is not None:
                embed = self.embed.cpu().numpy()
                for i in range(embed.shape[1]):
                    columns[f"embed_{i}"] = embed[:, i]
            return pd.DataFrame(columns)
        else:
            images = (self.image.cpu().numpy()*255).astype(np.uint8).squeeze(-1)
            columns : dict = {"image": [PIL.Image.fromarray(images)]}
            if self.label is not None: columns["label"] = [self.label.cpu().numpy()]
            if self.embed is not None:
                embed = self.embed.cpu().numpy()
                for i in range(embed.shape[1]):
                    columns[f"embed_{i}"] = [embed[:, i]]
            return pd.DataFrame(columns)

    def to_result(self) -> NestedResult:
        image = self.image.cpu().numpy()
        if image.ndim == 4:
            nrows = 1
            for n in range(1, int(math.sqrt(image.shape[0]))):
                if image.shape[0] % n == 0: nrows = n
            ncols = image.shape[0] // nrows
            image = image.reshape(ncols, nrows*image.shape[1], image.shape[2], image.shape[3])
            # move the ncols to the second dimension and reshape again
            image = image.transpose(1, 0, 2, 3).reshape(image.shape[1], ncols*image.shape[2], image.shape[3]) #
            if self.embed is None:
                return ImageResult(image)
            else:
                return {
                    "images": ImageResult(image),
                    "embeddings": TableResult(self.to_dataframe())
                }
        else:
            raise NotImplementedError()

    def with_values(self, sample: SampleValue, cond: CondValue | None = None) -> "Image":
        assert isinstance(sample, torch.Tensor)
        embed = self.embed if cond is None else cond
        return Image(sample, None, embed) # type: ignore

    @staticmethod
    def from_dataset(dataset: ds.Dataset) -> "ty.Iterator[Image]":
        mime_type = dataset.schema.metadata.get(b"mime_type").decode() if dataset.schema.metadata else ""
        has_class = "class" in mime_type
        has_tsne = "tsne" in mime_type
        labels, tsne = None, None
        for batch in dataset.to_batches():
            images = data_utils.decode_as_numpy(batch.column("image"), "image/encoded")
            images = torch.tensor(images.copy())
            if has_class:
                labels = data_utils.as_numpy(batch.column("class"))
                labels = torch.tensor(labels.copy())
            if has_tsne:
                tsne = data_utils.as_numpy(batch.column("tsne"))
                tsne = torch.tensor(tsne.copy())
            yield Image(images, labels, tsne)

pytree.register_pytree_node(
    Image, lambda image: ([image.image, image.label, image.embed], None),
    lambda children, aux: Image(children[0], children[1], children[2]) # type: ignore
)

def register_types(adapter: TorchAdapter[DataPoint]):
    adapter.register_type("data/trajectory", Trajectory.from_dataset)
    adapter.register_type("data/image", Image.from_dataset)
    adapter.register_type("data/point", Point.from_dataset)
    adapter.register_type("data/cond_point", Point.from_dataset)

class Visualizer(DataVisualizer):
    def __init__(self, override_mime_type: str | None = None):
        self.override_mime_type = override_mime_type

    def as_dataframe(self, split):
        adapter = TorchAdapter[DataPoint](override_mime_type=self.override_mime_type)
        register_types(adapter)
        dfs = []
        for batch in adapter.convert(split):
            dfs.append(batch.to_dataframe())
        return pd.concat(dfs)

class TsneTransform(DataTransform):
    def __init__(self, columns: list[str], dim: int = 2, perplexity: float = 30.0):
        self.columns = list(columns)
        self.components = dim
        self.perplexity = perplexity

    @property
    @ty.override
    def sha256(self) -> str:
        return hashlib.sha256(
            (f"tsne-{self.components}-{self.perplexity}-" + "-".join(self.columns)
        ).encode()).hexdigest()

    @ty.override
    def transform(self, data: Data, repo: DataRepository | None = None) -> Data:
        if repo is None: repo = DataRepository.default()
        sha = DataPipeline.transformed_sha256(data, self)
        transformed = repo.lookup(sha)
        if transformed is not None:
            return transformed
        # Read in all of the data
        # that we want to tsne
        tsne_data = []
        logger.info(f"Loading data for t-SNE")
        for name, split in data.splits():
            field_infos = [
                (field_name, split.schema.field(field_name).metadata.get(b"mime_type", b"field/unknown").decode())
                for field_name in split.schema.names
                if field_name in self.columns
            ] # get all of the fields that should be tsne'd
            for batch in split.to_batches():
                # flatten all of the
                tsne_data.append(np.concatenate([
                    data_utils.decode_as_numpy(batch.column(field), field_mime).reshape(batch.num_rows, -1)
                    for field, field_mime in field_infos
                ], axis=-1))
        tsne_data = np.concatenate(tsne_data, axis=0)
        # compute tsne for all of the data
        logger.info(f"Computing t-SNE on {tsne_data.shape}, {tsne_data.dtype}")
        tsne = TSNE(
            n_components=self.components,
            random_state=42, perplexity=self.perplexity
        )
        tsne_data = tsne.fit_transform(tsne_data)
        logger.info(f"Finished computing t-SNE: {tsne_data.shape}")

        tsne_field = pa.field("tsne", pa.list_(pa.float32(), tsne_data.shape[-1]),
            metadata={"mime_type": "vector/tsne"})
        curr_idx = 0
        with repo.init(sha) as writer:
            for name, split in data.splits():
                # compute the schema with the added "tsne" field
                schema = split.schema.append(pa.field("tsne", pa.list_(pa.float32(), tsne_data.shape[-1])))
                with writer.split(name) as split_writer:
                    for batch in split.to_batches():
                        # add an additional "tsne" column
                        batch_tsne = tsne_data[curr_idx:curr_idx+batch.num_rows]
                        curr_idx += batch.num_rows
                        batch = batch.append_column(tsne_field, data_utils.as_arrow_array(batch_tsne))
                        split_writer.write_batch(batch)
            return writer.close()

tsne_image = TsneTransform(["image"])
