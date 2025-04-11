from . import Experiment

import clearml
import typing as tp

####### ClearML Integration #######

class ClearMLExperiment(Experiment):
    def __init__(self, task: clearml.Task | None = None,
                *,
                 project_name : str | None = None):
        self.task = task if task else clearml.Task.init(
            project_name=project_name,
            auto_connect_streams={
                "stdout": False,
                "stderr": True,
                "logging": True
            }
        )
        self.task_logger = self.task.get_logger()
    
    def set_parameters(self, parameters: dict[str, tp.Any]):
        self.task.set_parameters(parameters)

    def log_metric(self, path: str, value: float,
                   series: str | None = None, step: int | None = None):
        self.task_logger.report_scalar(path, 
            series, value, iteration=step
        )

    def log_figure(self, path: str, figure : tp.Any, series: str | None = None, step: int | None = None):
        self.task_logger.report_plotly(
            path, series,
            figure=figure, iteration=step
        )
    
    def log_table(self, path: str, table: tp.Any, series: str | None = None, step: int | None = None):
        self.task_logger.report_table(
            path, series,
            table_plot=table, iteration=step
        )