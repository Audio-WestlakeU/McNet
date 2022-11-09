from typing import Dict, Optional
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_only


class MyLogger(TensorBoardLogger):
    """此类实现了val开头的metrics在输出到Tensorboard的时候，使用epoch作为step
    """

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None) -> None:
        for k, v in metrics.items():
            _my_step = step
            if k.startswith('val/'):  # use epoch for val metrics
                _my_step = int(metrics['epoch'])
            super().log_metrics(metrics={k: v}, step=_my_step)
