import imp
import os
from typing import List


from torch.utils.data import Dataset
import soundfile as sf
import time


class ValDataset(Dataset):

    def __init__(
        self,
        clean_dataset_dir: str = "~/simu-data/validation_dataset/clean/",
        noisy_dataset_dir: str = "~/simu-data/validation_dataset/noisy/",
        sr: int = 16000,
        ref_channel: int = 5,
        selected_channels: List[int] = [2, 3, 4, 5],
    ):
        super(ValDataset, self).__init__()

        clean_wav_list = os.listdir(os.path.abspath(os.path.expanduser(clean_dataset_dir)))
        noisy_wav_list = os.listdir(os.path.abspath(os.path.expanduser(noisy_dataset_dir)))
        clean_wav_list.sort()
        noisy_wav_list.sort()
        assert len(clean_wav_list) == len(noisy_wav_list)

        self.clean_wav_list = [os.path.join(os.path.abspath(os.path.expanduser(clean_dataset_dir)), cp) for cp in clean_wav_list]
        self.noisy_wav_list = [os.path.join(os.path.abspath(os.path.expanduser(noisy_dataset_dir)), np) for np in noisy_wav_list]

        self.length = len(noisy_wav_list)
        self.sr = sr
        self.ref_channel = ref_channel
        self.selected_channels = selected_channels

    def __len__(self):
        return self.length

    def __getitem__(self, index):

        noisy, sr = sf.read(self.noisy_wav_list[index], dtype='float32')
        noisy = noisy.T
        assert sr == self.sr
        # basename = self.noisy_wav_list[index].split('/')[-1].split('.')[-2].split('_')[-1]
        # clean_basename = '/mnt/home/yangyujie/real-data/validation_dataset/clean/clean_fileid_' + basename +'.wav'
        # clean, sr = sf.read(clean_basename, dtype='float32')
        clean, sr = sf.read(self.clean_wav_list[index], dtype='float32')
        clean = clean.T
        assert sr == self.sr
        clean_close = clean[-1, :]
        noisy = noisy[self.selected_channels, :]
        clean = clean[self.ref_channel, :]
        paras = {
            'index': index,
            'noisy_wav_path': self.noisy_wav_list[index],
            'clean_wav_path': self.clean_wav_list[index],
            'sr': self.sr,
            'selected_channels': self.selected_channels,
            'ref_channel': self.ref_channel,
            'clean_close': clean_close,
        }
        return noisy, clean, paras


if __name__ == '__main__':
    dataset = ValDataset()
    print(len(dataset))
    noisy, clean, paras = dataset[0]
    print(noisy.shape, clean.shape, paras['clean_close'].shape)
    del paras['clean_close']
    print(paras)
