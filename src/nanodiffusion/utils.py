from __future__ import annotations

import logging

import tempfile
import os
import json
import plotly.graph_objects
import pandas as pd
import rich
import typing as ty
import safetensors
import safetensors.torch
import torch
import numpy as np
import plotly

import rich._log_render
from rich.logging import RichHandler
from rich.progress import ProgressColumn
from rich.text import Text as RichText

from pathlib import Path

class Interval:
    def iterations(self, epoch_steps: int) -> int:
        raise NotImplementedError

    @ty.overload
    @staticmethod
    def to_iterations(interval: Interval | int, epoch_steps: int) -> int: ...

    @ty.overload
    @staticmethod
    def to_iterations(interval: None, epoch_steps: int) -> None: ...

    @staticmethod
    def to_iterations(interval: Interval | int | None, epoch_steps: int) -> int | None:
        if interval is None: return None
        if isinstance(interval, int): return interval
        return interval.iterations(epoch_steps)

class Epochs(Interval):
    def __init__(self, num: int):
        self.num = num
    def iterations(self, epoch_steps: int) -> int:
        return self.num*epoch_steps

class Iterations(Interval):
    def __init__(self, num: int):
        self.num = num
    def iterations(self, epoch_steps: int) -> int:
        return self.num

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

class MofNColumn(ProgressColumn):
    def __init__(self, min_width: int = 2):
        self.min_width = min_width
        super().__init__()

    def render(self, task) -> RichText:
        completed = int(task.completed)
        total = int(task.total) if task.total is not None else "?"
        total_width = len(str(total))
        total_padding = max(0,self.min_width - total_width)*" "
        total_width = max(self.min_width, total_width)
        return RichText(
            f"{completed:{total_width}d}/{total}{total_padding}",
            style="progress.percentage",
        )

class CustomLogRender(rich._log_render.LogRender):
    def __call__(self, *args, **kwargs):
        output = super().__call__(*args, **kwargs)
        if not self.show_path:
            output.expand = False
        return output

FORMAT = "%(name)s - %(message)s"

def setup_logging(show_path=False):
    # add_log_level("TRACE", logging.DEBUG - 5)
    logging.getLogger("nanodiffusion").setLevel(logging.INFO)
    if rich.get_console().is_jupyter:
        return rich.reconfigure(
            force_jupyter=False,
        )
    console = rich.get_console()
    handler = RichHandler(
        markup=True,
        rich_tracebacks=True,
        show_path=show_path,
        console=console
    )
    renderer = CustomLogRender(
        show_time=handler._log_render.show_time,
        show_level=handler._log_render.show_level,
        show_path=handler._log_render.show_path,
        time_format=handler._log_render.time_format,
        omit_repeated_times=handler._log_render.omit_repeated_times,
    )
    handler._log_render = renderer
    logging.basicConfig(
        level=logging.WARNING,
        format=FORMAT,
        datefmt="[%X]",
        handlers=[handler]
    )

# Fixes a bug in kaleido
# where the shebang is not set correctly
KALEIDO_PATCHED = False
def patch_kaleido():
    global KALEIDO_PATCHED
    if KALEIDO_PATCHED:
        return
    try:
        import kaleido.executable
        path = Path(kaleido.executable.__path__[0]) # type: ignore
        with open(path / "kaleido", "r") as f:
            first_line = f.readline()
            rest = f.read()
        if first_line != "#!/bin/bash\n":
            KALEIDO_PATCHED = True
            return
        with open(path / "kaleido", "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write(rest)
        KALEIDO_PATCHED = True
    except ImportError:
        pass
