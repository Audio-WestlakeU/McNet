import random
import os
import soundfile as sf
from time import time
from typing import Any, List, Tuple

import numpy as np
from scipy import signal

from torch.utils.data import Dataset
import torch
from torch import Tensor

from src.util.acoustic_utils import norm_amplitude, tailor_dB_FS, is_clipped, load_wav, subsample


class TrainDataset(Dataset):

    @staticmethod
    def collate_fn(batches: List[Any]):
        """把batches中的Tensor弄成batch，如果有其他参数则弄成list
        """
        mini_batch = []
        for x in zip(*batches):
            if isinstance(x[0], np.ndarray):
                x = [torch.tensor(x[i]) for i in range(len(x))]
            if isinstance(x[0], Tensor):
                x = torch.stack(x)
            mini_batch.append(x)
        return mini_batch

    def __init__(
        self,
        clean_dataset_dir: str = "~/simu-data/training_dataset/clean_speech/",
        noise_dataset_dir: str = "~/simu-data/training_dataset/noise_segment/",
        snr_range: Tuple[int, int] = (-5, 10),
        reverb_proportion: float = 0.75,
        silence_length: float = 0.2,
        target_dB_FS: int = -25,
        target_dB_FS_floating_value: int = 10,
        sub_sample_length: float = 3.072,
        sr: int = 16000,
        nchannels: int = 6,
        ref_channel: int = 5,
        selected_channels: List[int] = [2, 3, 4, 5],
    ):
        """
        Dynamic mixing for training

        Args:
            clean_dataset_limit:
            clean_dataset_offset:
            noise_dataset_limit:
            noise_dataset_offset:
            rir_dataset:
            rir_dataset_limit:
            rir_dataset_offset:
            snr_range:
            reverb_proportion:
            clean_dataset: scp file
            noise_dataset: scp file
            sub_sample_length:
            sr:
        """
        super().__init__()
        # acoustic args
        self.sr = sr

        # number of channels, the order of reference channel, the channels that have been chosen
        self.nchannels = nchannels
        self.ref_channel = ref_channel
        self.selected_channels = selected_channels

        # parallel args
        clean_dataset_list = os.listdir(os.path.abspath(os.path.expanduser(clean_dataset_dir)))
        noise_dataset_list = os.listdir(os.path.abspath(os.path.expanduser(noise_dataset_dir)))
        clean_dataset_list.sort()
        noise_dataset_list.sort()

        self.clean_dataset_list = [os.path.join(os.path.abspath(os.path.expanduser(clean_dataset_dir)), cp) for cp in clean_dataset_list]
        self.noise_dataset_list = [os.path.join(os.path.abspath(os.path.expanduser(noise_dataset_dir)), np) for np in noise_dataset_list]

        snr_list = self._parse_snr_range(snr_range)
        self.snr_list = snr_list

        assert 0 <= reverb_proportion <= 1, "reverberation proportion should be in [0, 1]"
        self.reverb_proportion = reverb_proportion
        self.silence_length = silence_length
        self.target_dB_FS = target_dB_FS
        self.target_dB_FS_floating_value = target_dB_FS_floating_value
        self.sub_sample_length = sub_sample_length

        self.length = len(self.clean_dataset_list)

    def __len__(self):
        return self.length

    @staticmethod
    def _random_select_from(dataset_list):
        return random.choice(dataset_list)

    def _select_noise_y(self, target_length):
        noise_y = np.zeros((self.nchannels, 0), dtype=np.float32)
        silence = np.zeros((self.nchannels, int(self.sr * self.silence_length)), dtype=np.float32)
        remaining_length = target_length

        while remaining_length > 0:
            noise_file = self._random_select_from(self.noise_dataset_list)
            noise_new_added,sr = sf.read(noise_file, dtype='float32')
            assert sr == self.sr
            noise_new_added = noise_new_added.T
            noise_y = np.append(noise_y, noise_new_added, axis=1)
            remaining_length -= np.shape(noise_new_added)[1]

            # 如果还需要添加新的噪声，就插入一个小静音段
            if remaining_length > 0:
                silence_len = min(remaining_length, np.shape(silence)[1])
                noise_y = np.append(noise_y, silence[:, :silence_len],axis=1)
                remaining_length -= silence_len

        if np.shape(noise_y)[1] > target_length:
            idx_start = np.random.randint(np.shape(noise_y)[1] - target_length)
            noise_y = noise_y[:, idx_start:idx_start + target_length]

        return noise_y

    @staticmethod
    def snr_mix(clean_y, noise_y, snr, target_dB_FS, target_dB_FS_floating_value, rir=None, eps=1e-6):
        """
        混合噪声与纯净语音，当 rir 参数不为空时，对纯净语音施加混响效果

        Args:
            clean_y: 纯净语音
            noise_y: 噪声
            snr (int): 信噪比
            target_dB_FS (int):
            target_dB_FS_floating_value (int):
            rir: room impulse response, None 或 np.array
            eps: eps

        Returns:
            (noisy_y，clean_y)
        """
        if rir is not None:
            if rir.ndim > 1:
                rir_idx = np.random.randint(0, rir.shape[0])
                rir = rir[rir_idx, :]

            clean_y = signal.fftconvolve(clean_y, rir)[:len(clean_y)]

        clean_y, _ = norm_amplitude(clean_y)
        clean_y, _, _ = tailor_dB_FS(clean_y, target_dB_FS)
        clean_rms = (clean_y**2).mean()**0.5

        noise_y, _ = norm_amplitude(noise_y)
        noise_y, _, _ = tailor_dB_FS(noise_y, target_dB_FS)
        noise_rms = (noise_y**2).mean()**0.5

        snr_scalar = clean_rms / (10**(snr / 20)) / (noise_rms + eps)
        noise_y *= snr_scalar
        noisy_y = clean_y + noise_y

        # Randomly select RMS value of dBFS between -15 dBFS and -35 dBFS and normalize noisy speech with that value
        noisy_target_dB_FS = np.random.randint(target_dB_FS - target_dB_FS_floating_value, target_dB_FS + target_dB_FS_floating_value)

        # 使用 noisy 的 rms 放缩音频
        noisy_y, _, noisy_scalar = tailor_dB_FS(noisy_y, noisy_target_dB_FS)
        clean_y *= noisy_scalar

        # 合成带噪语音的时候可能会 clipping，虽然极少
        # 对 noisy, clean_y, noise_y 稍微进行调整
        if is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)  # 相当于除以 1
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar

        return noisy_y, clean_y

    def __getitem__(self, index):
        clean_file = self.clean_dataset_list[index]
        #clean_y = load_wav(clean_file, sr=self.sr)
        clean_y, sr = sf.read(file=clean_file,dtype='float32')
        clean_y = clean_y.T
        assert sr == self.sr

        clean_y = subsample(clean_y, sub_sample_length=int(self.sub_sample_length * self.sr), nchannels=self.nchannels)

        noise_y = self._select_noise_y(target_length=clean_y.shape[1])
        assert np.shape(clean_y)[1] == noise_y.shape[1], f"Inequality: {len(clean_y)} {len(noise_y)}"

        snr = self._random_select_from(self.snr_list)

        noisy_y, clean_y = self.snr_mix(
            clean_y=clean_y,
            noise_y=noise_y,
            snr=snr,
            target_dB_FS=self.target_dB_FS,
            target_dB_FS_floating_value=self.target_dB_FS_floating_value,
        )

        noisy_y = noisy_y[self.selected_channels].astype(np.float32)
        clean_y = clean_y[self.ref_channel].astype(np.float32)
        
        return noisy_y, clean_y

    @staticmethod
    def _parse_snr_range(snr_range):
        assert len(snr_range) == 2, f"The range of SNR should be [low, high], not {snr_range}."
        assert snr_range[0] <= snr_range[-1], f"The low SNR should not larger than high SNR."

        low, high = snr_range
        snr_list = []
        for i in range(low, high + 1, 1):
            snr_list.append(i)

        return snr_list


if __name__ == "__main__":
    dataset = TrainDataset()
    print(len(dataset))
    noisy_y, clean_y = dataset[0]
    print(noisy_y.shape, clean_y.shape)
