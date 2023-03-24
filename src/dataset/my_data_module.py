from typing import Any, Callable, Dict, List, Tuple, Union
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset

from .train_dataset import TrainDataset
from .val_dataset import ValDataset
from .predict_dataset import PredictDataset
from .inf_dataset import InfDataset

from jsonargparse import lazy_instance  # 把默认参数注入command line interface（CLI）


class MyDataModule(LightningDataModule):

    def __init__(
        self,
        train_dataset: Union[Dataset, Tuple[Callable, Dict[str,Any]]] = lazy_instance(TrainDataset),
        val_dataset: Dataset = lazy_instance(ValDataset),
        inf_dataset: Dataset = lazy_instance(InfDataset),
        predict_dataset: Dataset = lazy_instance(PredictDataset),
        test_set: str = "test",
        batch_size: List[int] = [5, 1],
        num_workers: int = 15,
        # if pin_memory=True, will occupy a lot of memory & speed up
        pin_memory: bool = True,
        # prefetch how many samples, will increase the memory occupied when pin_memory=True
        prefetch_factor: int = 5,
        persistent_workers: bool = False,
    ):
        super().__init__()

        if isinstance(train_dataset, Dataset):
            self.train_dataset = train_dataset
        else:
            self.train_dataset = train_dataset[0](**train_dataset[1])
        self.val_dataset = val_dataset
        self.inf_dataset = inf_dataset
        self.predict_dataset = predict_dataset

        self.test_set = test_set

        self.batch_size_train = batch_size[0]
        self.batch_size_val = batch_size[1]
        self.batch_size_test = 1 if len(batch_size) == 2 else batch_size[2]
        self.batch_size_predict = 1 if len(batch_size) == 2 else batch_size[2]

        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        self.persistent_workers = persistent_workers

    def prepare_data(self):
        """prepare data function, will be called before setup
        """
        pass

    def setup(self, stage: str = None):
        """setup things before each stage

        Args:
            stage: fit, validation, or test
        """
        pass

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size_train,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=True,  # 一般train给shuffle=True，其他的给shuffle=False
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size_val,
            num_workers=self.num_workers,
            prefetch_factor=self.prefetch_factor,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            shuffle=False,  # val_dataloader没必要shuffle
        )

    def test_dataloader(self) -> DataLoader:
        prefetch_factor = 2

        if self.test_set == 'test':
            dataset = self.inf_dataset
        elif self.test_set == 'val':
            dataset = self.val_dataset
        else:  # train
            dataset = self.train_dataset

        return DataLoader(
            dataset,
            batch_size=self.batch_size_test,
            num_workers=0,
            prefetch_factor=prefetch_factor,
            shuffle=False,  # test_dataloader没必要shuffle
        )
    
    def predict_dataloader(self) -> DataLoader:
        prefetch_factor = 2
        dataset = self.predict_dataset
        return DataLoader(
            dataset,
            batch_size=self.batch_size_predict,
            num_workers=0,
            prefetch_factor=prefetch_factor,
            shuffle=False,
        )
