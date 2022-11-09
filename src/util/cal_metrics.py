from typing import Dict, List, Tuple, Union

from torch import Tensor

ALL_AUDIO_METRICS = ['SDR', 'SI_SDR', 'SI_SNR', 'SNR', 'NB_PESQ', 'WB_PESQ', 'STOI']

from torchmetrics.functional.audio import scale_invariant_signal_noise_ratio as si_snr
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import perceptual_evaluation_speech_quality as pesq
from torchmetrics.functional.audio import short_time_objective_intelligibility as stoi
from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from torchmetrics.functional.audio import signal_noise_ratio as snr


def cal_metrics_functional(
    metric_list: List[str],
    preds: Tensor,
    target: Tensor,
    original: Union[Tensor, Dict[str, Tensor]],
    fs: int,
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor], Dict[str, Tensor]]:
    preds_cpu = preds.detach().cpu()
    target_cpu = target.detach().cpu()

    if isinstance(original, Tensor):
        input_metrics = {}
        original_cpu = original.detach().cpu()
    else:
        input_metrics = original
        original_cpu = None

    metrics = {}
    imp_metrics = {}

    for m in metric_list:
        mname = m.lower()
        if m.upper() == 'SDR':
            metric_func = lambda: sdr(preds, target).mean().detach().cpu()
            input_metric_func = lambda: sdr(original, target).mean().detach().cpu()
        elif m.upper() == 'SI_SDR':
            metric_func = lambda: si_sdr(preds, target).mean().detach().cpu()
            input_metric_func = lambda: si_sdr(original, target).mean().detach().cpu()
        elif m.upper() == 'SI_SNR':
            metric_func = lambda: si_snr(preds, target).mean().detach().cpu()
            input_metric_func = lambda: si_snr(original, target).mean().detach().cpu()
        elif m.upper() == 'SNR':
            metric_func = lambda: snr(preds, target).mean().detach().cpu()
            input_metric_func = lambda: snr(original, target).mean().detach().cpu()
        elif m.upper() == 'NB_PESQ':
            metric_func = lambda: pesq(preds_cpu, target_cpu, fs, 'nb').mean()
            input_metric_func = lambda: pesq(original_cpu, target_cpu, fs, 'nb').mean()
        elif m.upper() == 'WB_PESQ':
            metric_func = lambda: pesq(preds_cpu, target_cpu, fs, 'wb').mean()
            input_metric_func = lambda: pesq(original_cpu, target_cpu, fs, 'wb').mean()
        elif m.upper() == 'STOI':
            metric_func = lambda: stoi(preds_cpu, target_cpu, fs).mean()
            input_metric_func = lambda: stoi(original_cpu, target_cpu, fs).mean()
        else:
            raise ValueError('Unkown audio metric ' + m)

        if m.upper() == 'WB_PESQ' and fs == 8000:
            continue  # Note: have narrow band (nb) mode only when sampling rate is 8000Hz

        metrics[m.lower()] = metric_func()
        if 'input_' + mname not in input_metrics.keys():
            input_metrics['input_' + mname] = input_metric_func()
        imp_metrics[mname + '_i'] = metrics[mname] - input_metrics['input_' + mname]

    return metrics, input_metrics, imp_metrics
