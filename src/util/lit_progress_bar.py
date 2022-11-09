"""进度条：显示进度+每个epoch的metrics总结（仅val的是平均值，train的似乎是最后一个值；等lightning更新后使用最新版本替换此实现）
"""

from typing import Any, Optional, Union
from pytorch_lightning.callbacks import RichProgressBar

import sys
from torch import Tensor
from pytorch_lightning import Trainer
from rich.console import Console, RenderableType
from rich.console import Console, RenderableType
from rich.progress import BarColumn, Progress, ProgressColumn, Task, TaskID, TextColumn
from rich.progress_bar import ProgressBar
from rich.style import Style
from rich.text import Text
import math
from datetime import timedelta
from dataclasses import dataclass


class CustomBarColumn(BarColumn):
    """Overrides ``BarColumn`` to provide support for dataloaders that do not define a size (infinite size)
    such as ``IterableDataset``."""

    def render(self, task: "Task") -> ProgressBar:
        """Gets a progress bar widget for a task."""
        return ProgressBar(
            total=max(0, task.total),
            completed=max(0, task.completed),
            width=None if self.bar_width is None else max(1, self.bar_width),
            pulse=not task.started or not math.isfinite(task.remaining),
            animation_time=task.get_time(),
            style=self.style,
            complete_style=self.complete_style,
            finished_style=self.finished_style,
            pulse_style=self.pulse_style,
        )


@dataclass
class CustomInfiniteTask(Task):
    """Overrides ``Task`` to define an infinite task.

    This is useful for datasets that do not define a size (infinite size) such as ``IterableDataset``.
    """

    @property
    def time_remaining(self) -> Optional[float]:
        return None


class CustomProgress(Progress):
    """Overrides ``Progress`` to support adding tasks that have an infinite total size."""

    def add_task(
        self,
        description: str,
        start: bool = True,
        total: float = 100.0,
        completed: int = 0,
        visible: bool = True,
        **fields: Any,
    ) -> TaskID:
        if not math.isfinite(total):
            task = CustomInfiniteTask(
                self._task_index,
                description,
                total,
                completed,
                visible=visible,
                fields=fields,
                _get_time=self.get_time,
                _lock=self._lock,
            )
            return self.add_custom_task(task)
        return super().add_task(description, start, total, completed, visible, **fields)

    def add_custom_task(self, task: CustomInfiniteTask, start: bool = True):
        with self._lock:
            self._tasks[self._task_index] = task
            if start:
                self.start_task(self._task_index)
            new_task_index = self._task_index
            self._task_index = TaskID(int(self._task_index) + 1)
        self.refresh()
        return new_task_index


class CustomTimeColumn(ProgressColumn):

    # Only refresh twice a second to prevent jitter
    max_refresh = 0.5

    def __init__(self, style: Union[str, Style]) -> None:
        self.style = style
        super().__init__()

    def render(self, task) -> Text:
        elapsed = task.finished_time if task.finished else task.elapsed
        remaining = task.time_remaining
        elapsed_delta = "-:--:--" if elapsed is None else str(timedelta(seconds=int(elapsed)))
        remaining_delta = "-:--:--" if remaining is None else str(timedelta(seconds=int(remaining)))
        return Text(f"{elapsed_delta} • {remaining_delta}", style=self.style)


class BatchesProcessedColumn(ProgressColumn):

    def __init__(self, style: Union[str, Style]):
        self.style = style
        super().__init__()

    def render(self, task) -> RenderableType:
        total = task.total if task.total != float("inf") else "--"
        return Text(f"{int(task.completed)}/{total}", style=self.style)


class ProcessingSpeedColumn(ProgressColumn):

    def __init__(self, style: Union[str, Style]):
        self.style = style
        super().__init__()

    def render(self, task) -> RenderableType:
        task_speed = f"{task.speed:>.2f}" if task.speed is not None else "0.00"
        return Text(f"{task_speed}it/s", style=self.style)


class MetricsTextColumn(ProgressColumn):
    """A column containing text."""

    def __init__(self, trainer, style):
        self._trainer = trainer
        self._tasks = {}
        self._current_task_id = 0
        self._metrics = {}
        self._style = style
        super().__init__()

    def update(self, metrics):
        # Called when metrics are ready to be rendered.
        # This is to prevent render from causing deadlock issues by requesting metrics
        # in separate threads.
        self._metrics = metrics

    def render(self, task) -> Text:
        from pytorch_lightning.trainer.states import TrainerFn

        if self._trainer.state.fn != TrainerFn.FITTING or self._trainer.sanity_checking:
            return Text("")
        if self._trainer.training and task.id not in self._tasks:
            self._tasks[task.id] = "None"
            if self._renderable_cache:
                self._tasks[self._current_task_id] = self._renderable_cache[self._current_task_id][1]
            self._current_task_id = task.id
        if self._trainer.training and task.id != self._current_task_id:
            return self._tasks[task.id]
        _text = ""

        for k, v in self._metrics.items():
            _text += f"{k}: {round(v, 3) if isinstance(v, float) else v} "
        return Text(_text, justify="left", style=self._style)


class LitProgressBar(RichProgressBar):
    """A progress bar prints metrics at the end of each epoch
    """

    def _init_progress(self, trainer):
        if self.is_enabled and (self.progress is None or self._progress_stopped):
            self._reset_progress_bar_ids()
            self._console: Console = Console(force_terminal=True, no_color=True, width=200)
            self._console.clear_live()
            self._metric_component = MetricsTextColumn(trainer, self.theme.metrics)
            self.progress = CustomProgress(
                *self.configure_columns(trainer),
                self._metric_component,
                refresh_per_second=self.refresh_rate_per_second,
                disable=self.is_disabled,
                console=self._console,
            )
            self.progress.start()
            # progress has started
            self._progress_stopped = False

    def on_validation_epoch_end(self, trainer: Trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        sys.stdout.flush()
        if trainer.is_global_zero:
            metrics = trainer.logged_metrics
            infos = f"Epoch {trainer.current_epoch} metrics: "
            for k, v in metrics.items():
                value = v
                if isinstance(v, Tensor):
                    value = v.item()
                if isinstance(value, float):
                    infos += k + f"={value:.4f}  "
                else:
                    infos += k + f"={value}  "
            if len(metrics) > 0:
                sys.stdout.write(f'{infos}\n')
            sys.stdout.flush()
