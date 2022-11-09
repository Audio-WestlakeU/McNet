"""
Command Line Interface: 命令行接口，命令行指定配置文件，修改参数等
"""

import os

os.environ["OMP_NUM_THREADS"] = str(8)  # 限制进程数量，防止CPU被占满

from src.model.mc_net import *
from src.util.my_save_config_callback import MySaveConfigCallback as SaveConfigCallback
from src.util.my_rich_progress_bar import MyRichProgressBar as RichProgressBar
from src.util.my_logger import MyLogger
from pytorch_lightning.cli import (LightningArgumentParser, LightningCLI)
from pytorch_lightning.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint, ModelSummary)
import pytorch_lightning as pl
from src.dataset.my_data_module import MyDataModule
import torch
from jsonargparse import lazy_instance


torch.backends.cuda.matmul.allow_tf32 = True  # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
torch.backends.cudnn.allow_tf32 = True  # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.

# from pytorch_lightning.loggers import TensorBoardLogger

TensorBoardLogger = MyLogger


class MyCLI(LightningCLI):

    def add_arguments_to_parser(self, parser: LightningArgumentParser) -> None:
        """1. 添加默认参数，使得可以通过print_config生成默认的参数文件
            2. 添加需要使用的插件
        """

        # 设置默认参数
        parser.set_defaults({"trainer.strategy": "ddp_find_unused_parameters_false",
                             "trainer.max_epochs": 200,
                             "trainer.check_val_every_n_epoch": 10,
                             })
        parser.set_defaults(
            {'model.freq': lazy_instance(RNN_FC, input_size=6*2, output_size=256, hidden_size=128, num_layers=1, bidirectional=True, act_funcs=['', ''],use_FC=True),
             'model.narr': lazy_instance(RNN_FC, input_size=256, output_size=64, hidden_size=256, num_layers=1, bidirectional=False, act_funcs=['', ''],use_FC=True),
             'model.sub':  lazy_instance(RNN_FC, input_size=64*(2*7+1), output_size=16, hidden_size=128, num_layers=1, bidirectional=False, act_funcs=['', ''],use_FC=True),
             'model.full':  lazy_instance(RNN_FC, input_size=16*257, output_size=257*2, hidden_size=512, num_layers=1, bidirectional=False, act_funcs=['', ''],use_FC=True),
             'model.io': lazy_instance(McNetIO, loss_func=y_mse),
             },
        )

        # RichProgressBar
        parser.add_lightning_class_args(RichProgressBar, nested_key='progress_bar')
        if pl.__version__.startswith('1.5.'):
            parser.set_defaults({
                "progress_bar.refresh_rate_per_second": 1,
            })
        else:
            parser.set_defaults({
                "progress_bar.refresh_rate": 1,
            })
            parser.set_defaults({
                "progress_bar.console_kwargs": {
                    "force_terminal": True,
                    "no_color": True,
                    "width": 200,
                },
            })

        # ModelCheckpoint
        parser.add_lightning_class_args(ModelCheckpoint, "model_checkpoint")
        model_checkpoint_defaults = {
            "model_checkpoint.filename": "epoch{epoch}_criteria{val/criteria:.2f}_sdr{val/sdr:.2f}",
            "model_checkpoint.monitor": "val/criteria",
            "model_checkpoint.mode": "max",
            "model_checkpoint.every_n_epochs": 1,
            "model_checkpoint.save_top_k": 20,
            "model_checkpoint.auto_insert_metric_name": False,
            "model_checkpoint.save_last": True
        }
        parser.set_defaults(model_checkpoint_defaults)

        # LearningRateMonitor
        parser.add_lightning_class_args(LearningRateMonitor, "learning_rate_monitor")
        learning_rate_monitor_defaults = {
            "learning_rate_monitor.logging_interval": "epoch",
        }
        parser.set_defaults(learning_rate_monitor_defaults)

        # ModelSummary
        parser.add_lightning_class_args(ModelSummary, 'model_summary')
        model_summary_defaults = {
            "model_summary.max_depth": -1,
        }
        parser.set_defaults(model_summary_defaults)

        return super().add_arguments_to_parser(parser)

    def before_fit(self):
        """实现的功能是：如果fit时的ckpt_path和trainer.resume_from_checkpoint被指定且为last.ckpt，则在原训练目录接着训练；否则在新目录训练
        """
        resume_from_checkpoint: str = self.config['fit']['trainer']["resume_from_checkpoint"] or self.config['fit']['ckpt_path']
        if resume_from_checkpoint is not None and resume_from_checkpoint.endswith('last.ckpt'):
            # log in same dir
            # resume_from_checkpoint example: /mnt/home/quancs/projects/NBSS_pmt/logs/NBSS_ifp/version_29/checkpoints/last.ckpt
            resume_from_checkpoint = os.path.normpath(resume_from_checkpoint)
            splits = resume_from_checkpoint.split(os.path.sep)
            version = int(splits[-3].replace('version_', ''))
            save_dir = os.path.sep.join(splits[:-3])
            self.trainer.logger = TensorBoardLogger(save_dir=save_dir, name="", version=version, default_hp_metric=False)
        else:
            model_name = type(self.model).__name__
            self.trainer.logger = TensorBoardLogger('logs/', name=model_name, default_hp_metric=False)

    def after_fit(self):
        """训练完成之后，取出最好的checkpoint进行测试
        """
        if self.trainer.limit_test_batches is not None and self.trainer.limit_test_batches <= 0:
            return
        # test
        torch.set_num_interop_threads(5)
        torch.set_num_threads(5)
        resume_from_checkpoint = self.trainer.checkpoint_callback.best_model_path
        if resume_from_checkpoint is None or resume_from_checkpoint == "":
            if self.trainer.is_global_zero:
                print("no checkpoint found, so test is ignored")
            return
        epoch = os.path.basename(resume_from_checkpoint).split('_')[0]
        write_dir = os.path.dirname(os.path.dirname(resume_from_checkpoint))
        exp_save_path = os.path.normpath(write_dir + '/' + epoch + '_' + self.config['fit']['data']["test_set"] + '_set')

        # comment the following code to disable the test after fit
        import torch.distributed as dist
        if self.trainer.is_global_zero:
            self.trainer.logger = TensorBoardLogger(exp_save_path, name="", default_hp_metric=False)
            versions = [self.trainer.logger.version]
        else:
            versions = [None]
        if self.trainer.world_size > 1:
            dist.broadcast_object_list(versions)
            self.trainer.logger = TensorBoardLogger(exp_save_path, name="", version=versions[0], default_hp_metric=False)
        self.trainer.test(ckpt_path=resume_from_checkpoint, datamodule=self.datamodule)
        self.after_test()

    def before_test(self):
        """功能：在测试时，调整测试目录为：训练目录/epoch_xxx_test_set/version_x

        Raises:
            Exception: _description_
        """
        torch.set_num_interop_threads(5)
        torch.set_num_threads(5)
        if self.config['test']['ckpt_path'] != None:
            ckpt_path = self.config['test']['ckpt_path']
        else:
            raise Exception('You should give --ckpt_path if you want to test')
        epoch = os.path.basename(ckpt_path).split('_')[0]
        write_dir = os.path.dirname(os.path.dirname(ckpt_path))
        exp_save_path = os.path.normpath(write_dir + '/' + epoch + '_' + self.config['test']['data']["test_set"] + '_set')

        self.trainer.logger = TensorBoardLogger(exp_save_path, name='', default_hp_metric=False)

    def after_test(self):
        """测试完成之后移除测试目录下tensorboard的相关文件
        """
        if not self.trainer.is_global_zero:
            return
        import fnmatch
        files = fnmatch.filter(os.listdir(self.trainer.log_dir), 'events.out.tfevents.*')
        for f in files:
            os.remove(self.trainer.log_dir + '/' + f)
            print('tensorboard log file for test is removed: ' + self.trainer.log_dir + '/' + f)


if __name__ == '__main__':
    cli = MyCLI(McNet, MyDataModule, seed_everything_default=True, save_config_overwrite=True, save_config_callback=SaveConfigCallback)
