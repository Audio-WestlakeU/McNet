import warnings
import torch.nn as nn
import json
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import time

import pytorch_lightning as pl
import soundfile as sf
import torch
from pandas import DataFrame
from src.util.cal_metrics import cal_metrics_functional
from src.util.git_tools import tag_and_log_git_status
from src.util.my_json_encoder import MyJsonEncoder
from src.util.acoustic_utils import get_complex_ideal_ratio_mask, icIRM
from torch import Tensor
from torch.nn import Module
from torch.nn.functional import pad
from torchmetrics.functional.audio import perceptual_evaluation_speech_quality as pesq
from torchmetrics.functional.audio.snr import signal_noise_ratio as snr
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
from torchmetrics.functional.audio import short_time_objective_intelligibility as stoi
from torchmetrics.functional.audio import signal_distortion_ratio as sdr
from pytorch_lightning.utilities.rank_zero import rank_zero_info


def neg_snr(prediction: Tensor, target: Tensor) -> Tensor:
    return -snr(preds=prediction, target=target)


def neg_si_sdr(prediction: Tensor, target: Tensor) -> Tensor:
    return -si_sdr(preds=prediction, target=target)


def y_mse(prediction: Tensor, target: Tensor) -> Tensor:
    mse_loss = torch.nn.MSELoss(reduction='mean')
    return mse_loss(prediction, target)


def stft_y_mse(prediction: Tensor, target: Tensor) -> Tensor:
    mse_loss = torch.nn.MSELoss(reduction='mean')
    prediction = torch.view_as_real(prediction)
    target = torch.view_as_real(target)
    return mse_loss(prediction, target)


def cumulative_normalization(original_signal_mag: Tensor, sliding_window_len: int = 192) -> Tensor:
    alpha = (sliding_window_len - 1) / (sliding_window_len + 1)
    eps = 1e-10
    mu = 0
    mu_list = []
    batch_size, frame_num, freq_num = original_signal_mag.shape
    for frame_idx in range(frame_num):
        if frame_idx < sliding_window_len:
            alp = torch.min(torch.tensor([(frame_idx - 1) / (frame_idx + 1), alpha]))
            mu = alp * mu + (1 - alp) * torch.mean(original_signal_mag[:, frame_idx, :], dim=-1).reshape(batch_size, 1)
        else:
            current_frame_mu = torch.mean(original_signal_mag[:, frame_idx, :], dim=-1).reshape(batch_size, 1)
            mu = alpha * mu + (1 - alpha) * current_frame_mu
        mu_list.append(mu)

    XrMM = torch.stack(mu_list, dim=-1).permute(0, 2, 1).reshape(batch_size, frame_num, 1, 1)
    return XrMM


class McNetIO(Module):
    """网络的input，output以及loss相关的部分

    当前实现是：
        输入：全部通道的STFT系数；输出：参考通道的STFT系数。
    """

    def __init__(
        self,
        selected_channels: List[int] = [2, 3, 4, 5],
        ref_channel: int = 5,
        loss_func: Callable = neg_si_sdr,
        ft_len: int = 512,
        ft_hop: int = 256,
        sliding_window_len: int = 192,
        use_cumulative_normalization: bool = False,
    ) -> None:
        super().__init__()
        self.register_buffer("window", torch.hann_window(ft_len))

        self.ft_len = ft_len
        self.ft_hop = ft_hop
        self.window = torch.hann_window(self.ft_len)

        self.selected_channels = selected_channels
        self.use_cumulative_normalization = use_cumulative_normalization
        self.sliding_window_len = sliding_window_len
        self.ref_chn_idx = selected_channels.index(ref_channel)

        self.loss_func = loss_func
        self._loss_name = loss_func.__name__

        self.freq_num = self.ft_len // 2 + 1

    def prepare_input(self, x: Tensor, *args, **kwargs) -> Dict[str, Any]:
        batch_size, chn_num, time = x.shape

        # stft x
        x = x.reshape((batch_size * chn_num, time))
        X = torch.stft(x, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, return_complex=True)  # type:ignore
        X = X.reshape((batch_size, chn_num, X.shape[-2], X.shape[-1]))  # (batch, channel, freq, time frame)
        X = X.permute(0, 3, 2, 1)  # (batch, time frame, freq, channel)

        # normalization by using ref_channel
        frame_num, freq_num = X.shape[1], X.shape[2]
        Xr = X[..., self.ref_chn_idx].clone()  # copy
        if self.use_cumulative_normalization == False:
            XrMM = torch.abs(Xr).mean(dim=(1, 2)).reshape(batch_size, 1, 1, 1)  # Xr_magnitude_mean: mean of the magnitude of the ref channel of X
        else:
            XrMM = cumulative_normalization(original_signal_mag=torch.abs(Xr), sliding_window_len=self.sliding_window_len)

        X /= (XrMM + 1e-8)
        input = torch.view_as_real(X).reshape(batch_size, frame_num, freq_num, chn_num * 2)
        XrMag = torch.abs(X[:, :, :, self.ref_chn_idx]).unsqueeze(-1)
        XMag = torch.abs(X)
        return {'input': input, "X": X, "XrMM": XrMM, "XrMag": XrMag, "XMag": XMag, "original_time_len": time}

    def prepare_target(self, x: Tensor, yr: Tensor, XrMM: Tensor, *args, **kwargs) -> Any:
        """prepare target for loss function
        """
        yr_norm = yr / XrMM.reshape(XrMM.shape[0], 1)
        return yr_norm

    def prepare_prediction(self, X: Tensor, output: Tensor, XrMM: Tensor, original_time_len: int, *args, **kwargs) -> Tensor:
        """prepare prediction from the output of network for loss function
        """
        batch_size, frame_num, freq_num, chn_num = X.shape
        output = torch.view_as_complex(output.reshape(batch_size, frame_num, freq_num, 2))  # [B,T,F]
        output = output.permute(0, 2, 1)  # [B, F, T]

        pred = torch.istft(output, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, length=original_time_len)  # [B, t]  type:ignore
        return pred

    def prepare_time_domain(self, x: Tensor, prediction: Tensor, XrMM: Tensor, original_time_len: int, *args, **kwargs) -> Tensor:
        """prepare time domain prediction
        """
        return prediction * XrMM.reshape(XrMM.shape[0], 1)  # 此处再乘上系数，prepare_prediction处乘上XrMM会使得不同的句子具有不同的权重

    def loss(self, prediction: Tensor, target: Tensor, reduce_batch: bool = False, *args, **kwargs) -> Tensor:
        """loss for prediction and target
        """
        if reduce_batch:
            return self.loss_func(prediction=prediction, target=target).mean()
        else:
            return self.loss_func(prediction=prediction, target=target)

    @property
    def loss_name(self) -> str:
        return self._loss_name


class CCIO(McNetIO):

    def prepare_target(self, x: Tensor, yr: Tensor, XrMM: Tensor, *args, **kwargs) -> Any:
        """prepare target for loss function
        """
        target = torch.stft(yr, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, return_complex=True)  # type:ignore # (batch, freq, time frame)
        target /= (XrMM.reshape(XrMM.shape[0], 1, 1) + 1e-8)  # normalization as prediction, (batch, freq, time frame)
        return target

    def prepare_prediction(self, X: Tensor, output: Tensor, XrMM: Tensor, original_time_len: int, *args, **kwargs) -> Tensor:
        """prepare prediction from the output of network for loss function
        """
        batch_size, frame_num, freq_num, chn_num = X.shape
        output = output.reshape(batch_size, frame_num, freq_num, 2).permute(0, 2, 1, 3)  # [B,F,T,2]
        prediction = torch.view_as_complex(output)  # [B,F,T]
        return prediction

    def prepare_time_domain(self, x: Tensor, prediction: Tensor, XrMM: Tensor, original_time_len: int, *args, **kwargs) -> Tensor:  # type:ignore
        """prepare time domain prediction
        """
        prediction = prediction * XrMM.reshape(XrMM.shape[0], 1, 1)
        wav = torch.istft(prediction, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, length=original_time_len)  # [B, t]  type:ignore
        return wav


class cIRMIO(McNetIO):

    def prepare_target(self, x: Tensor, yr: Tensor, XrMM: Tensor, *args, **kwargs) -> Any:
        """prepare target for loss function
        """
        if len(x.shape) == 3:
            xr = x[:, self.ref_chn_idx, :]
        else:
            assert len(x.shape) == 2
            xr = x
        assert len(yr.shape) == 2
        stft_xr = torch.stft(xr, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, return_complex=False)
        stft_yr = torch.stft(yr, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, return_complex=False)
        assert stft_xr.dim() == stft_yr.dim()
        target = get_complex_ideal_ratio_mask(noisy_complex_tensor=stft_xr, clean_complex_tensor=stft_yr)
        target = torch.view_as_complex(target)
        return target

    def prepare_prediction(self, X: Tensor, output: Tensor, XrMM: Tensor, original_time_len: int, *args, **kwargs) -> Tensor:
        """prepare prediction from the output of network for loss function
        """
        batch_size, frame_num, freq_num, chn_num = X.shape
        output = output.reshape(batch_size, frame_num, freq_num, 2).permute(0, 2, 1, 3)  # [B,F,T,2]
        prediction = torch.view_as_complex(output)  # [B,F,T]
        return prediction

    def prepare_time_domain(self, x: Tensor, prediction: Tensor, XrMM: Tensor, original_time_len: int, *args, **kwargs) -> Tensor:
        if len(x.shape) == 3:
            xr = x[:, self.ref_chn_idx, :]
        else:
            assert len(x.shape) == 2
            xr = x
        stft_xr = torch.stft(xr, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, return_complex=False)
        prediction = torch.view_as_real(prediction)
        prediction_icIRM = icIRM(pred_cIRM=prediction, noisy_complex_tensor=stft_xr)
        wav = torch.istft(prediction_icIRM, n_fft=self.ft_len, hop_length=self.ft_hop, window=self.window, win_length=self.ft_len, length=original_time_len)
        return wav


class RNN_FC(nn.Module):

    def __init__(
            self,
            input_size: int,
            output_size: int,
            hidden_size: int,
            num_layers: int = 2,
            bidirectional: bool = True,
            act_funcs: Tuple[str, str] = ('SiLU', ''),
            use_FC: bool = True,
    ):
        super().__init__()

        # Sequence layer
        self.sequence_model = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # If batch_first is true, the shape of input tensor should be [batch_size,time_step,feature]. The output is the same.
            bidirectional=bidirectional,
        )
        self.sequence_model.flatten_parameters()

        # Fully connected layer
        self.use_FC = use_FC
        if self.use_FC:
            if bidirectional:
                self.fc_output_layer = nn.Linear(hidden_size * 2, output_size)
            else:
                self.fc_output_layer = nn.Linear(hidden_size, output_size)

        # Activation function layer
        self.act_funcs = []
        for act_func in act_funcs:
            if act_func == 'SiLU' or act_func == 'swish':
                self.act_funcs.append(nn.SiLU())
            elif act_func == 'ReLU':
                self.act_funcs.append(nn.ReLU())
            elif act_func == 'Tanh':
                self.act_funcs.append(nn.Tanh())
            elif act_func == None or act_func == '':
                self.act_funcs.append(None)  # type:ignore
            else:
                raise NotImplementedError(f"Not implemented activation function {act_func}")

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: [B, T, Feature]
        Returns:
            [B, T, Feature]
        """
        o, _ = self.sequence_model(x)
        if self.act_funcs[0] != None:
            o = self.act_funcs[0](o)
        if self.use_FC:
            o = self.fc_output_layer(o)
            if self.act_funcs[1] != None:
                o = self.act_funcs[1](o)
        return o


class McNet(pl.LightningModule):
    """控制训练，测试，推理的各个方面
    """

    def __init__(
        self,
        io: McNetIO,
        freq: Optional[Union[Module, List[Module]]] = None,
        narr: Optional[Union[Module, List[Module]]] = None,
        sub: Optional[Union[Module, List[Module]]] = None,
        full: Optional[Union[Module, List[Module]]] = None,
        order: List[str] = ['freq', 'narr', 'sub', 'full'],
        sub_freqs: Union[int, Tuple[int, int]] = (3, 2),  # (N_1=3, N_2=2)
        look_past_and_ahead: Tuple[int, int] = (5, 0),
        learning_rate: float = 0.001,
        optimizer_kwargs: Dict[str, Any] = dict(),
        lr_scheduler: str = 'ExponentialLR',
        lr_scheduler_kwargs: Dict[str, Any] = {'gamma': 0.992},
        # if exp_name==notag, then git version control is disabled
        exp_name: str = "exp",
        use_dense_net: bool = False,
        use_time_domain_loss: bool = False,
    ):
        super().__init__()
        self.freq = freq if not isinstance(freq, List) else nn.Sequential(*freq)
        self.narr = narr if not isinstance(narr, List) else nn.Sequential(*narr)
        self.sub = sub if not isinstance(sub, List) else nn.Sequential(*sub)
        self.full = full if not isinstance(full, List) else nn.Sequential(*full)
        self.io = io
        self.order = order
        self.use_dense_net = use_dense_net
        self.use_time_domain_loss = use_time_domain_loss
        # save all the hyperparameters to self.hparams
        self.save_hyperparameters(ignore=['freq', 'narr', 'sub', 'full', 'io'])

    def on_train_start(self):
        rank_zero_info('order: ' + str(self.hparams.order) + ', sub_freqs: ' + str(self.hparams.sub_freqs))
        if self.current_epoch == 0:
            if self.trainer.is_global_zero and hasattr(self.logger, 'log_dir') and 'notag' not in self.hparams.exp_name:
                # add git tags for better change tracking
                # note: if change self.logger.log_dir to self.trainer.log_dir, the training will stuck on multi-gpu training
                tag_and_log_git_status(self.logger.log_dir + '/git.out', self.logger.version, self.hparams.exp_name, model_name=type(self).__name__)

    def forward(self, x: Tensor) -> Tuple[Tensor, Dict[str, Any]]:  # type:ignore
        data = self.io.prepare_input(x)
        i: Tensor = data['input']  # shape [B,T,F,H]
        B, T, F, H = i.shape
        index = 1
        input_dict = dict()  # type:ignore

        for curr in self.order:
            if self.use_dense_net:
                for idx in range(2, index):
                    i = torch.concat([i, input_dict[idx - 1]], dim=-1)
                B, T, F, H = i.shape
            if not curr.startswith('sub2') and not curr.startswith('sub3') and not curr.startswith('full4freq'):
                if curr.endswith('+X'):  # 等价于 +
                    i = torch.concat([i, data['input']], dim=-1)
                    curr = curr.replace('+X', '')
                elif curr.endswith('+XrMag'):
                    i = torch.concat([i, data['XrMag']], dim=-1)
                    curr = curr.replace('+XrMag', '')

            # 子带部分的hidden是否除以频率的个数
            reduce_by_num_freqs = False
            if curr == 'sub_':
                reduce_by_num_freqs = True
                curr = 'sub'

            if curr == 'freq':  # 沿频率的序列，[BT, F, H]
                i = i.reshape(B * T, F, -1)
                i = self.freq(i).reshape(B, T, F, -1)  # type:ignore # to shape [B,T,F,H]
            elif curr == 'narr':  # 沿时间的窄带，[BF, T, H]
                i = i.permute(0, 2, 1, 3).reshape(B * F, T, -1)
                i = self.narr(i)  # type:ignore
                i = i.reshape(B, F, T, -1).permute(0, 2, 1, 3)  # to shape [B,T,F,H]
            elif curr == 'sub':  # 沿时间的子带，[BF, T, H']
                i = i.permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)  # to shape [BT, H, F, 1], 1 is for unfold module (4D supported only)
                i = torch.concat([i[:, :, :self.hparams.sub_freqs, :], i, i[:, :, -self.hparams.sub_freqs:, :]], dim=2)  # type:ignore # freqs: -3 -2 -1 0 ... 256 0 1 2
                i = torch.nn.functional.unfold(i, kernel_size=(self.hparams.sub_freqs * 2 + 1, 1))  # type:ignore # shape [BT, H'=H*2S+1, F]
                i = i.reshape(B, T, -1, F).permute(0, 3, 1, 2).reshape(B * F, T, -1)
                if reduce_by_num_freqs:
                    i = i / (self.hparams.sub_freqs * 2 + 1)  # type:ignore
                i = self.sub(i)  # type:ignore
                i = i.reshape(B, F, T, -1).permute(0, 2, 1, 3)  # to shape [B,T,F,H]
            elif curr.startswith('sub2'):  # 子带部分输入=【上层输出对应频带的embedding，原始信号拼接的子带】
                if curr.endswith('+X'):
                    X = data['input'].permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)  # to shape [BT, H, F, 1], 1 is for unfold module (4D supported only)
                else:
                    assert curr.endswith('+XrMag'), curr
                    X = data['XrMag'].permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)  # to shape [BT, H, F, 1], 1 is for unfold module (4D supported only)
                if self.hparams.sub_freqs != 0:
                    X = torch.concat([X[:, :, :self.hparams.sub_freqs, :], X, X[:, :, -self.hparams.sub_freqs:, :]], dim=2)  # type:ignore # freqs: -3 -2 -1 0 ... 256 0 1 2
                    Xsub = torch.nn.functional.unfold(X, kernel_size=(self.hparams.sub_freqs * 2 + 1, 1))  # type:ignore # shape [BT, H'=H*2S+1, F]，原始信号拼接的子带
                    i = i.permute(0, 1, 3, 2).reshape(B * T, -1, F)  # to shape [BT, H, F]
                else:
                    Xsub = X.reshape(B * T, -1, F)
                    i = i.permute(0, 1, 3, 2).reshape(B * T, -1, F)
                i = torch.concat([i, Xsub], dim=1)
                i = i.reshape(B, T, -1, F).permute(0, 3, 1, 2).reshape(B * F, T, -1)
                i = self.sub(i)  # type:ignore
                i = i.reshape(B, F, T, -1).permute(0, 2, 1, 3)  # to shape [B,T,F,H]
            elif curr.startswith('sub3'):
                assert len(self.hparams.sub_freqs) == 2
                if curr.endswith('+X'):
                    X = data['input'].permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)  # to shape [BT, H, F, 1], 1 is for unfold module (4D supported only)
                else:
                    assert curr.endswith('+XrMag'), curr
                    X = data['XrMag'].permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)  # to shape [BT, H, F, 1], 1 is for unfold module (4D supported only)
                if self.hparams.sub_freqs[0] != 0:
                    X = torch.concat([X[:, :, :self.hparams.sub_freqs[0], :], X, X[:, :, -self.hparams.sub_freqs[0]:, :]], dim=2)  # type:ignore # freqs: -3 -2 -1 0 ... 256 0 1 2
                    Xsub = torch.nn.functional.unfold(X, kernel_size=(self.hparams.sub_freqs[0] * 2 + 1, 1))  # type:ignore # shape [BT, H'=H*2S+1, F]，原始信号拼接的子带
                else:
                    Xsub = X.reshape(B * T, -1, F)
                i = i.permute(0, 1, 3, 2).reshape(B * T, -1, F, 1)  # to shape [BT, H, F, 1]
                i = torch.concat([i[:, :, :self.hparams.sub_freqs[1], :], i, i[:, :, -self.hparams.sub_freqs[1]:, :]], dim=2)
                i = torch.nn.functional.unfold(i, kernel_size=(self.hparams.sub_freqs[1] * 2 + 1, 1))
                i = torch.concat([i, Xsub], dim=1)
                i = i.reshape(B, T, -1, F).permute(0, 3, 1, 2).reshape(B * F, T, -1)
                i = self.sub(i)  # type:ignore
                i = i.reshape(B, F, T, -1).permute(0, 2, 1, 3)  # to shape [B,T,F,H]

            elif curr.startswith('full2'):  # 沿时间的所有通道的单通道全频带放在第一层: [BH,T,F]
                i = data['XMag'].permute(0, 3, 1, 2)  #[B,H,T,F]
                B, H, T, F = i.shape
                i = i.reshape(B * H, T, F)  #[BH,T,F]
                i = self.full(i)  # type:ignore
                i = i.reshape(B, H, T, F, -1).permute(0, 2, 3, 1, 4)
                i = i.reshape(B, T, F, -1)  #to shape [B,T,F,H]
            elif curr.startswith('full3freq'):  #沿着频率轴的所有通道的单通道全频带放在第一层：
                if curr.endswith('-allch'):
                    i = data['XMag'].permute(0, 2, 3, 1)  #[B,F,H,T] here H=the number of channels
                    B, F, H, T = i.shape
                    i = torch.nn.functional.pad(i, pad=self.hparams.look_past_and_ahead, mode='constant', value=0)
                    i = i.reshape(B * F * H, 1, -1, 1)  # unfold module (4D supported only)
                    i = torch.nn.functional.unfold(i, kernel_size=(self.hparams.look_past_and_ahead[0] + self.hparams.look_past_and_ahead[1] + 1, 1))  #[B*F*H,H'=(T_look_ahead+T_look_past+1)*1,T]
                    i = i.reshape(B, F, H, -1, T).permute(0, 2, 4, 1, 3).reshape(B * H * T, F, -1)
                    i = self.full(i)  # type:ignore
                    i = i.reshape(B, H, T, F, -1).permute(0, 2, 3, 1, 4)
                    i = i.reshape(B, T, F, -1)  #to shape [B,T,F,H]
                else:
                    i = data['XrMag'].permute(0, 2, 3, 1)  #[B,F,H,T]
                    B, F, H, T = i.shape
                    i = torch.nn.functional.pad(i, pad=self.hparams.look_past_and_ahead, mode='constant', value=0)
                    i = i.reshape(B * F, H, -1, 1)  # unfold module (4D supported only)
                    i = torch.nn.functional.unfold(i, kernel_size=(self.hparams.look_past_and_ahead[0] + self.hparams.look_past_and_ahead[1] + 1, 1))  #[B*F,H'=(T_look_ahead+T_look_past+1)*H,T]
                    i = i.reshape(B, F, -1, T).permute(0, 3, 1, 2).reshape(B * T, F, -1)  #[B*T,F,H']
                    i = self.full(i).reshape(B, T, F, -1)  # type:ignore
            elif curr.startswith('fullfreq'):
                i = i.reshape(B * T, F, -1)
                i = self.full(i).reshape(B, T, F, -1)  # type:ignore # to shape [B,T,F,H]
            elif curr.startswith('full4freq'):
                i = i.reshape(B * T, F, -1)
                if curr.endswith('+XrMag'):
                    XrMag = data['XrMag'].permute(0, 2, 3, 1)  #[B,F,H,T]
                    XrMag = torch.nn.functional.pad(XrMag, pad=self.hparams.look_past_and_ahead, mode='constant', value=0)
                    XrMag = XrMag.reshape(B * F, 1, -1, 1)
                    XrMag = torch.nn.functional.unfold(XrMag,
                                                       kernel_size=(self.hparams.look_past_and_ahead[0] + self.hparams.look_past_and_ahead[1] + 1, 1))  #[B*F,H'=(T_look_ahead+T_look_past+1)*1,T]
                    XrMag = XrMag.reshape(B, F, -1, T).permute(0, 3, 1, 2).reshape(B * T, F, -1)  #[B*T,F,H']
                    i = torch.cat([i, XrMag], dim=-1)
                i = self.full(i).reshape(B, T, F, -1)  # type:ignore # to shape [B,T,F,H]
            else:  # 沿时间的全频带：[B, T, FH]
                assert curr == 'full', curr
                i = i.reshape(B, T, -1)
                i = self.full(i)  # type:ignore
                i = i.reshape(B, T, F, -1)
            input_dict[index] = i
            index += 1

        data['output'] = i
        prediction = self.io.prepare_prediction(**data)
        return prediction, data

    def training_step(self, batch, batch_idx):
        x, yr = batch  # x [B, C, T]; yr [B, T]
        prediction, data = self.forward(x)
        target = self.io.prepare_target(x=x, yr=yr, **data)
        if self.use_time_domain_loss == False:
            loss = self.io.loss(prediction=prediction, target=target, reduce_batch=True)
        else:
            yr_hat = self.io.prepare_time_domain(x=x, prediction=prediction, **data)
            loss = self.io.loss(prediction=yr_hat, target=yr, reduce_batch=True)

        self.log('train/' + self.io.loss_name, loss, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x, yr, paras = batch  # x [B, C, T]; yr [B, T]

        prediction, data = self.forward(x)
        target = self.io.prepare_target(x=x, yr=yr, **data)
        yr_hat = self.io.prepare_time_domain(x=x, prediction=prediction, **data)
        if self.use_time_domain_loss == False:
            loss = self.io.loss(prediction=prediction, target=target, reduce_batch=True)
        else:
            loss = self.io.loss(prediction=yr_hat, target=yr, reduce_batch=True)

        assert yr_hat.shape[1] == yr.shape[1]

        SDR = sdr(preds=yr_hat, target=yr).mean()
        CRITERIA = SDR

        if self.trainer.current_epoch % 10 == 9:
            try:
                WB_PESQ = pesq(preds=yr_hat, target=yr, fs=16000, mode='wb').mean()
                SISDR = si_sdr(preds=yr_hat, target=yr).mean()
                NB_PESQ = pesq(preds=yr_hat, target=yr, fs=16000, mode='nb').mean()
                STOI = stoi(preds=yr_hat, target=yr, fs=16000).mean()
                self.log('val/sisdr', SISDR, batch_size=x.shape[0], sync_dist=True)
                self.log('val/nb_pesq', NB_PESQ, batch_size=x.shape[0], sync_dist=True)
                self.log('val/stoi', STOI, batch_size=x.shape[0], sync_dist=True)
                self.log('val/wb_pesq', WB_PESQ, batch_size=x.shape[0], sync_dist=True)
            except Exception as e:
                warnings.warn(str(e))

        self.log('val/' + self.io.loss_name, loss, batch_size=x.shape[0], sync_dist=True)
        self.log('val/sdr', SDR, batch_size=x.shape[0], sync_dist=True)
        self.log('val/criteria', CRITERIA, batch_size=x.shape[0], logger=False, sync_dist=True)
        return loss

    def on_test_epoch_start(self):
        # 测试开始时，创建保存目录
        self.exp_save_path = os.path.join(self.trainer.logger.log_dir, "examples")
        os.makedirs(self.exp_save_path, exist_ok=True)

    def test_epoch_end(self, results):
        """Called by PytorchLightning automatically at the end of test epoch"""
        import torch.distributed as dist

        # collect results from other gpus if world_size > 1
        if self.trainer.world_size > 1:
            results_list = [None for obj in results]
            dist.all_gather_object(results_list, results)  # gather results from all gpus
            # merge them
            exist = set()
            results = []
            for rs in results_list:
                if rs == None:
                    continue
                for r in rs:
                    if r['wav_name'] not in exist:
                        results.append(r)
                        exist.add(r['wav_name'])

        # save collected data on 0-th gpu
        if self.trainer.is_global_zero:
            # Tensor to list or number
            for r in results:
                for key, val in r.items():
                    if isinstance(val, Tensor):
                        if val.numel() == 1:
                            r[key] = val.item()
                        else:
                            r[key] = val.detach().cpu().numpy().tolist()

            # save
            import datetime
            x = datetime.datetime.now()
            dtstr = x.strftime('%Y%m%d_%H%M%S.%f')
            path = os.path.join(self.trainer.logger.log_dir, 'results_{}.json'.format(dtstr))
            # write results to json
            f = open(path, 'w', encoding='utf-8')
            json.dump(results, f, indent=4, cls=MyJsonEncoder)
            f.close()
            # write mean to json
            df = DataFrame(results)
            df.mean(numeric_only=True).to_json(os.path.join(self.trainer.logger.log_dir, 'results_mean.json'), indent=4)

    def test_step(self, batch, batch_idx):
        x, yr, paras = batch  # x [B, C, T]; yr [B, T]
        prediction, data = self.forward(x)
        yr_hat = self.io.prepare_time_domain(x=x, prediction=prediction, **data)
        target = self.io.prepare_target(x=x, yr=yr, **data)
        assert yr_hat.shape[1] == yr.shape[1]

        if self.use_time_domain_loss == False:
            loss = self.io.loss(prediction=prediction, target=target, reduce_batch=True)
        else:
            loss = self.io.loss(prediction=yr_hat, target=yr, reduce_batch=True)

        # calculate loss
        self.log('test/' + self.io.loss_name, loss, logger=False, batch_size=x.shape[0])

        # write examples
        assert x.shape[0] == 1, "The batch size of inference stage must 1."
        wav_name = paras['wav_name'][0] if 'wav_name' in paras else str(paras['index'].cpu().item()) + '.wav'
        sr = paras['sr'][0]
        result_dict = {'id': batch_idx, 'wav_name': wav_name, self.io.loss_name: loss.item()}

        if abs(yr_hat).any() > 1:
            print(f"Warning: enhanced is not in the range [-1, 1], {wav_name}")

        # calculate metrics, input_metrics, improved_metrics
        metric_list = ['SDR', 'SI_SDR', 'NB_PESQ', 'WB_PESQ', 'STOI']
        metrics, input_metrics, imp_metrics = cal_metrics_functional(metric_list=metric_list, preds=yr_hat, target=yr, original=x[:, self.io.ref_chn_idx, :], fs=sr.item())
        for key, val in metrics.items():
            self.log('test/' + key, val, logger=False, batch_size=x.shape[0])
            result_dict[key] = val

        result_dict.update(input_metrics)
        result_dict.update(imp_metrics)

        def write_wav(wav_path: str, wav: torch.Tensor, norm_to: torch.Tensor = None):
            # make sure wav don't have illegal values (abs greater than 1)
            abs_max = torch.max(torch.abs(wav))
            if norm_to:
                wav = wav / abs_max * norm_to
            if abs_max > 1:
                wav /= abs_max
            sf.write(wav_path, wav.detach().cpu().numpy(), samplerate=sr.item())

        write_wav(wav_path=self.exp_save_path + "/" + f"{wav_name}", wav=yr_hat[0])
        if 'clean_close' in paras:
            del paras['clean_close']

        result_dict['paras'] = paras
        return result_dict

    def predict_step(self, batch: Tensor, batch_idx: Optional[int] = None, dataloader_idx: Optional[int] = None) -> Tensor:
        """predict
        Args:
            batch: x. shape of x [B, C, T]

        Returns:
            Tensor: y_hat, shape [B, T]; prediction_cIRM_mask,shape;data
        """
        x = batch
        prediction, data = self.forward(x)
        yr_hat = self.io.prepare_time_domain(x=x, prediction=prediction, **data)
        return yr_hat

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate, **self.hparams.optimizer_kwargs)

        if self.hparams.lr_scheduler != None and len(self.hparams.lr_scheduler) > 0:
            lr_scheduler = getattr(torch.optim.lr_scheduler, self.hparams.lr_scheduler)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': lr_scheduler(optimizer, **self.hparams.lr_scheduler_kwargs),
                    'monitor': 'val/' + self.io.loss_name,
                }
            }
        else:
            return optimizer


if __name__ == "__main__":
    with torch.no_grad():
        io = McNetIO([0, 1, 2, 3, 4, 5], ref_channel=4, loss_func=stft_y_mse, ft_len=512, ft_hop=256, sliding_window_len=192, use_cumulative_normalization=True)
        model = McNet(
            freq=RNN_FC(input_size=6 * 2, output_size=64, hidden_size=128, num_layers=1, bidirectional=True, act_funcs=('', 'ReLU'), use_FC=True),
            narr=RNN_FC(input_size=76, output_size=64, hidden_size=256, num_layers=1, bidirectional=False, act_funcs=('', 'ReLU'), use_FC=True),
            sub=RNN_FC(input_size=327, output_size=64, hidden_size=384, num_layers=1, bidirectional=False, act_funcs=('', 'ReLU'), use_FC=True),
            full=RNN_FC(input_size=70, output_size=2, hidden_size=128, num_layers=1, bidirectional=True, act_funcs=('', ''), use_FC=True),
            order=['freq', 'narr+X', 'sub3+XrMag', 'full4freq+XrMag'],
            sub_freqs=(3, 2),
            look_past_and_ahead=(5, 0),
            io=io,
        )
        noisy = torch.rand(1, 4, 32000)

        yr_hat = model.predict_step(noisy)
        print(yr_hat.shape)
