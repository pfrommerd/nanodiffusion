import abc
import pandas as pd
import typing as tp
import numpy as np

from .. import config, field, utils

class Dummy: ...

# PlotlyFigure = Dummy
# MplFigure = Dummy
try:
    from plotly.graph_objects import Figure as PlotlyFigure
except ImportError:
    pass
try:
    from matplotlib.figure import Figure as MplFigure
except ImportError:
    pass

class Experiment(abc.ABC):
    def log(self, result: "NestedResult",
                  path: str | None = None,
                  step: int | None = None):
        for k, v in utils.flatten_items(result):
            if isinstance(v, Result):
                v.log(self, path=k, step=step)
            else:
                raise TypeError(f"Unsupported type {type(v)} for logging")

    @abc.abstractmethod
    def log_metric(self, path: str, value: float,
                   series: str |None = None, step: int | None = None): ...

    @abc.abstractmethod
    def log_figure(self, path: str, figure: PlotlyFigure | MplFigure | dict,
                   series: str | None = None, step: int | None = None): ...

    @abc.abstractmethod
    def log_table(self, path: str, table: pd.DataFrame,
                  series: str | None = None, step: int | None = None): ...

class ConsoleExperiment(Experiment):
    def __init__(self, logger, series_intervals={}):
        self.logger = logger
        self.series_intervals = series_intervals

    def log_metric(self, path: str, value: float,
                   series: str | None = None, step: int | None = None):
        console_path = path.replace("/", ".")
        if step is not None:
            interval = self.series_intervals.get(series, 1)
            if step % interval == 0:
                self.logger.info(f"{step} - {console_path}: {value} ({series})")

    def log_figure(self, path: str, figure : tp.Any, 
                   series: str | None = None, step: int | None = None):
        pass

    def log_table(self, path: str, table: pd.DataFrame,
                  series: str | None = None, step: int | None = None):
        pass

class CombinedExperiment(Experiment):
    def __init__(self, *experiments: Experiment):
        self.experiments = experiments

    def log_metric(self, path: str, value: float, 
                   series : str | None = None, step: int | None = None):
        for experiment in self.experiments:
            experiment.log_metric(path, value, series=series, step=step) 
    
    def log_figure(self, path: str, figure : tp.Any,
                     series: str | None = None, step: int | None = None):
          for experiment in self.experiments:
                experiment.log_figure(path, figure, series=series, step=step)
    
    def log_table(self, path: str, table: pd.DataFrame,
                     series: str | None = None, step: int | None = None):
          for experiment in self.experiments:
                experiment.log_table(path, table, series=series, step=step)

class Result(abc.ABC):
    @abc.abstractmethod
    def log(self, experiment: Experiment, path: str, step: int | None): ...

NestedResult = dict[str, "NestedResult"] | Result

class Metric(Result):
    def __init__(self, series: str | None, value: float):
        self.series = series
        self.value = value

    def log(self, experiment: Experiment, path: str, step: int | None):
        experiment.log_metric(path, self.series, self.value, step=step)

class Table(Result):
    def __init__(self, dataframe: pd.DataFrame):
        self.dataframe = dataframe
    
    def log(self, experiment: Experiment, path: str, step: int | None):
        experiment.log_table(path, self.dataframe, step=step)

class Figure(Result):
    def __init__(self, figure):
        self.figure = figure
    
    def log(self, experiment: Experiment, path: str, step: int | None):
        experiment.log_figure(path, self.figure, step=step)


@config
class ExperimentConfig:
    project: str | None = None

    console: bool = True
    # The interval only applies to 
    # metrics with the train/ prefix
    console_intervals: dict[str, int] = field(default_factory=lambda: {
        "train": 100,
        "test": 1
    })

    wandb: bool = False
    clearml: bool = False
    cometml : bool = False

    def create(self, logger):
        experiments = []
        if self.console:
            experiments.append(ConsoleExperiment(
                logger=logger, series_intervals=self.console_intervals
            ))
        if self.clearml:
            from .clearml import ClearMLExperiment
            experiments.append(ClearMLExperiment(project_name=self.project))
        return CombinedExperiment(*experiments)