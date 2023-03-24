import os
from typing import List

from torch.utils.data import Dataset
import soundfile as sf


class PredictDataset(Dataset):

    def __init__(
        self,
        noisy_dataset_dir: str = "~/simu-data/test_dataset/noisy",
        sr: int = 16000,
        ref_channel: int = 5,
        selected_channels: List[int] = [2, 3, 4, 5],
    ):
        super(PredictDataset, self).__init__()
        noisy_wav_list = os.listdir(os.path.abspath(os.path.expanduser(noisy_dataset_dir)))
        noisy_wav_list.sort()
        
        self.noisy_dataset_file_name = noisy_dataset_dir.split('/')[-2]
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
        
        noisy = noisy[self.selected_channels, :]
        wav_name = os.path.basename(self.noisy_wav_list[index])
        paras = {
            'index': index,
            'noisy_wav_path': self.noisy_wav_list[index],
            'sr': self.sr,
            'selected_channels': self.selected_channels,
            'ref_channel': self.ref_channel,
            'wav_name': wav_name,
            'noisy_dataset_file_name':self.noisy_dataset_file_name,
        }
        return noisy, paras


if __name__ == '__main__':
    dataset = PredictDataset()
    print(len(dataset))
    noisy, paras = dataset[0]
    print(noisy.shape, paras)
