"""
Utility functions for loading PNAS Leaves images into TorchVision dataloaders

Created by: Monday April 12th, 2021
Author: Jacob A Rose



"""

from torchmetrics import Accuracy
from flash import Task
from torch import nn, optim, Generator
import flash
from flash.vision import ImageClassificationData, ImageClassifier
from torchvision import models
import pytorch_lightning as pl
from typing import List, Callable, Dict, Union, Type, Optional
from pytorch_lightning.callbacks import Callback
from contrastive_learning.data.pytorch.flash.process import Preprocess, Postprocess

# TODO (Jacob): Hardcode the mean & std for PNAS, Extant Leaves, Imagenet, etc.. for standardization across lab

from torchvision import transforms
from pathlib import Path
from glob import glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.datasets import ImageFolder
from typing import Callable, Optional, Any
from munch import Munch

# log the in- and output histograms of LightningModule's `forward`
# monitor = ModuleDataMonitor()

from .common import SubsetImageDataset, seed_worker


available_datasets = {"PNAS_family_100_512": "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_512",
                      "PNAS_family_100_1024": "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100_1024"}
default_name = "PNAS_family_100_512"


from torchvision.datasets.vision import VisionDataset

# class LeavesDataset(VisionDataset):
# class LeavesDataset(ImageFolder):

#     splits_on_disk : Tuple[str]= ("train", "test")
    
#     def __init__(
#             self,
#             name: str=default_name,
#             split: str="train",
#             **kwargs: Any
#             ) -> None:

#         assert split in self.splits_on_disk
#         assert name in available_datasets, f"{name} is not in the set of available datasets. Please try one of the following: \n{available_datasets.keys()}"
        
#         self.name = name
#         self.split = split
#         self.dataset_dir = Path(available_datasets[name])
#         self.split_dir = self.dataset_dir / self.split
        
#         super().__init__(root=self.split_dir,
#                          **kwargs)







class PNASLightningDataModule(pl.LightningDataModule):
    
    available_datasets = available_datasets

    preprocess_cls = Preprocess
    postprocess_cls = Postprocess
    worker_init_fn=seed_worker
    
    image_size = 224
    target_size = (224, 224)
    image_buffer_size = 32
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    def __init__(self,
                 name: str=None,
                 batch_size: int=32,
                 val_split: float=0.2,
                 num_workers=0,
                 pin_memory=False,
                 shuffle_train: bool=True,
                 seed: int=None):
        
        super().__init__()
        
        assert ((val_split >= 0) and (val_split <= 1)), "[!] val_split should be in the range [0, 1]."
        self.val_split = val_split
        
        assert (name in self.available_datasets) | (name is None)
        self.name = name or default_name
        
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        self.val_split = val_split
        self.shuffle_train = shuffle_train
        self.seed = seed
        
        self.train_dir = Path(self.available_datasets[self.name], 'train')
        self.test_dir = Path(self.available_datasets[self.name], 'test')
        
        self._initialized = False        

    def init_dataset_stage(self,
                           stage: str='fit',
                           train_transform: Optional[Callable] = None,
                           eval_transform: Optional[Callable] = None,
                           target_transform: Optional[Callable] = None):
        
        
        train_transform = train_transform # or self.default_train_transforms()
        eval_transform = eval_transform # or self.default_eval_transforms()
        target_transform = target_transform
        
        if stage == 'fit' or stage is None:
            self.train_dataset = ImageFolder(root=self.train_dir, transform=train_transform, target_transform=target_transform)
            self.classes = self.train_dataset.classes

            val_split = self.val_split
            num_train = len(self.train_dataset)
            split_idx = (int(np.floor((1-val_split) * num_train)), int(np.floor(val_split * num_train)))
            if self.seed is None:
                generator = None
            else:
                generator = Generator().manual_seed(self.seed)

            train_data, val_data = random_split(self.train_dataset, 
                                                split_idx, # [split_idx[0], split_idx[1]], 
                                                generator=generator)

            self.train_dataset = SubsetImageDataset(train_data.dataset, train_data.indices)
            self.val_dataset = SubsetImageDataset(val_data.dataset, val_data.indices)
            
        elif stage == 'test' or stage is None:
            self.test_dataset = ImageFolder(root=self.test_dir, transform=eval_transform, target_transform=target_transform)
        
        
    def setup(self,
              stage: str=None,
              train_transform: Optional[Callable] = None,
              eval_transform: Optional[Callable] = None,
              target_transform: Optional[Callable] = None
              ):
        if stage == 'fit' or stage is None:
            self.init_dataset_stage(stage='fit',
                                    train_transform=train_transform,
                                    eval_transform=eval_transform,
                                    target_transform=target_transform)
        if stage == 'test' or stage is None:
            self.init_dataset_stage(stage='test',
                                    eval_transform=eval_transform,
                                    target_transform=target_transform)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                                  num_workers=self.num_workers,
                                  pin_memory=self.pin_memory,
                                  batch_size=self.batch_size,
                                  shuffle=self.shuffle_train)
        return train_loader
        
    def val_dataloader(self):
        val_loader = DataLoader(self.val_dataset,
                                num_workers=self.num_workers,
                                pin_memory=self.pin_memory,
                                batch_size=self.batch_size*10,
                                shuffle=False)
        return val_loader

    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset,
                                 num_workers=self.num_workers,
                                 pin_memory=self.pin_memory,
                                 batch_size=self.batch_size*10,
                                 shuffle=False)
        return test_loader
    
    
    @staticmethod
    def default_train_transforms():
        image_size = PNASLightningDataModule.image_size
        image_buffer_size = PNASLightningDataModule.image_buffer_size
        mean, std = PNASLightningDataModule.mean, PNASLightningDataModule.std
        return transforms.Compose([transforms.Resize(image_size+image_buffer_size),
                                   transforms.RandomHorizontalFlip(p=0.5),
                                   transforms.RandomCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
#         return {
#             "pre_tensor_transform": transforms.Compose([transforms.Resize(image_size+image_buffer_size),
#                                                         transforms.RandomHorizontalFlip(p=0.5),
#                                                         transforms.RandomCrop(image_size)]),
#             "to_tensor_transform": transforms.ToTensor(),
#             "post_tensor_transform": transforms.Normalize(mean, std),
#         }
    ##############
    @staticmethod
    def default_eval_transforms():
        image_size = PNASLightningDataModule.image_size
        image_buffer_size = PNASLightningDataModule.image_buffer_size
        mean, std = PNASLightningDataModule.mean, PNASLightningDataModule.std
        return transforms.Compose([transforms.Resize(image_size+image_buffer_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])

    
    
    
    
#     def train_dataset(self,
#                        train_transform: Optional[Callable] = None,
#                        eval_transform: Optional[Callable] = None,
#                        target_transform: Optional[Callable] = None):
#         self.setup(stage='fit',
#                    train_transform=train_transform,
#                    eval_transform=eval_transform,
#                    target_transform=target_transform)
# #         self.init_dataset_stage(stage='fit',
# #                                 train_transform=train_transform,
# #                                 eval_transform=eval_transform,
# #                                 target_transform=target_transform)
        
#     def val_dataset(self,
#                      eval_transform: Optional[Callable] = None,
#                      target_transform: Optional[Callable] = None):
#         self.setup(stage='fit',
#                    eval_transform=eval_transform,
#                    target_transform=target_transform)

#         self.init_dataset_stage(stage='fit',
#                                 eval_transform=eval_transform,
#                                 target_transform=target_transform)

#     def test_dataset(self,
#                       eval_transform: Optional[Callable] = None,
#                       target_transform: Optional[Callable] = None):
#         self.init_dataset_stage(stage='test',
#                                 eval_transform=eval_transform,
#                                 target_transform=target_transform)
    
    
    
    
    
    
    
    
    #         return {
#             "pre_tensor_transform": transforms.Compose([transforms.Resize(image_size+image_buffer_size),
#                                                         transforms.CenterCrop(image_size)]),
#             "to_tensor_transform": transforms.ToTensor(),
#             "post_tensor_transform": transforms.Normalize(mean, std)
#         }

        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
#     # we define a separate DataLoader for each of train/val/test
#     def train_dataloader(self):
#         train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size)
#         return train_dataloader

#     def val_dataloader(self):
#         val_dataloader = DataLoader(self.val_dataset, batch_size=10 * self.batch_size)
#         return val_dataloader

#     def test_dataloader(self):
#         test_dataloader = DataLoader(self.test_dataset, batch_size=10 * self.batch_size)
#         return test_dataloader
        
        
        

        
        
        
        
        
        
        
        
        
        
        
        
        
        
#         self._initialized = True
#         return self.train_dataset, self.val_dataset, self.test_dataset

#     def train_dataloader(self):
#         train_loader = DataLoader(self.train_dataset,
#                                   num_workers=self.num_workers,
#                                   pin_memory=self.pin_memory,
#                                   batch_size=self.batch_size,
#                                   shuffle=self.shuffle_train)
#         return train_loader
        
#     def val_dataloader(self):
#         val_loader = DataLoader(self.val_dataset,
#                                 num_workers=self.num_workers,
#                                 pin_memory=self.pin_memory,
#                                 batch_size=self.batch_size*10,
#                                 shuffle=False)
#         return val_loader

#     def test_dataloader(self):
#         test_loader = DataLoader(self.test_dataset,
#                                  num_workers=self.num_workers,
#                                  pin_memory=self.pin_memory,
#                                  batch_size=self.batch_size*10,
#                                  shuffle=False)
#         return test_loader
#############################################
    
#     def init_dataloaders(self, 
#                          num_workers: int=0,
#                          batch_size: int=32,
#                          pin_memory: bool=False,
#                          shuffle_train: bool=True):
        
#         if not self._initialized:
#             self.init_datasets()
        
#         self.train_loader = DataLoader(self.train_dataset,
#                                   num_workers=num_workers,
#                                   pin_memory=pin_memory,
#                                   batch_size=batch_size,
#                                   shuffle=shuffle_train)
#         self.val_loader = DataLoader(self.val_dataset,
#                                 num_workers=num_workers,
#                                 pin_memory=pin_memory,
#                                 batch_size=batch_size,
#                                 shuffle=False)
#         self.test_loader = DataLoader(self.test_dataset,
#                                  num_workers=num_workers,
#                                  pin_memory=pin_memory,
#                                  batch_size=batch_size,
#                                  shuffle=False)
        
#         return self.train_loader, self.val_loader, self.test_loader
        

#########################################
#########################################
#########################################
#########################################

# class MNISTDataModule(pl.LightningDataModule):
# class PNASImageDataModule(ImageClassificationData):
#     """Data module for image classification tasks."""
#     preprocess_cls = Preprocess
#     postprocess_cls = Postprocess
    
#     image_size = 224
#     target_size = (224, 224)
#     image_buffer_size = 32
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]

#     def __init__(self,
#                  name: str=None,
#                  batch_size: int=128,
#                  val_split: float=0.2,
#                  num_workers=0,
#                  pin_memory=False,
#                  shuffle_train: bool=True,
#                  seed: int=None):
        
#         super().__init__()
#         self.name = name or default_name
#         self.batch_size = batch_size
#         self.val_split = val_split
#         self.num_workers = num_workers
#         self.pin_memory = pin_memory
#         self.shuffle_train = shuffle_train
#         self.seed = seed

#     def prepare_data(self):
#         self.data = PNASLeaves(name=self.name, 
#                                val_split=self.val_split)
        
#     def setup(self, stage: str=None):
        
#         self.data.setup(stage=stage,
#                         train_transform=self.default_train_transforms(),
#                         eval_transform=self.default_val_transforms())
#         if stage == 'fit' or stage is None:
#             self.train_dataset, self.val_dataset = self.data.train_dataset, self.data.val_dataset
#         if stage == 'test' or stage is None:
#             self.test_dataset = self.data.test_dataset


#     def train_dataloader(self):
#         train_loader = DataLoader(self.train_dataset,
#                                   num_workers=self.num_workers,
#                                   pin_memory=self.pin_memory,
#                                   batch_size=self.batch_size,
#                                   shuffle=self.shuffle_train)
#         return train_loader
        
#     def val_dataloader(self):
#         val_loader = DataLoader(self.val_dataset,
#                                 num_workers=self.num_workers,
#                                 pin_memory=self.pin_memory,
#                                 batch_size=self.batch_size*10,
#                                 shuffle=False)
#         return val_loader

#     def test_dataloader(self):
#         test_loader = DataLoader(self.test_dataset,
#                                  num_workers=self.num_workers,
#                                  pin_memory=self.pin_memory,
#                                  batch_size=self.batch_size*10,
#                                  shuffle=False)
#         return test_loader
    
    
#     @staticmethod
#     def default_train_transforms():
#         image_size = PNASImageDataModule.image_size
#         image_buffer_size = PNASImageDataModule.image_buffer_size
#         mean, std = PNASImageDataModule.mean, PNASImageDataModule.std
#         return transforms.Compose([transforms.Resize(image_size+image_buffer_size),
#                                    transforms.RandomHorizontalFlip(p=0.5),
#                                    transforms.RandomCrop(image_size),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize(mean, std)])
# #         return {
# #             "pre_tensor_transform": transforms.Compose([transforms.Resize(image_size+image_buffer_size),
# #                                                         transforms.RandomHorizontalFlip(p=0.5),
# #                                                         transforms.RandomCrop(image_size)]),
# #             "to_tensor_transform": transforms.ToTensor(),
# #             "post_tensor_transform": transforms.Normalize(mean, std),
# #         }
#     ##############
#     @staticmethod
#     def default_val_transforms():
#         image_size = PNASImageDataModule.image_size
#         image_buffer_size = PNASImageDataModule.image_buffer_size
#         mean, std = PNASImageDataModule.mean, PNASImageDataModule.std
#         return transforms.Compose([transforms.Resize(image_size+image_buffer_size),
#                                    transforms.CenterCrop(image_size),
#                                    transforms.ToTensor(),
#                                    transforms.Normalize(mean, std)])
# #         return {
# #             "pre_tensor_transform": transforms.Compose([transforms.Resize(image_size+image_buffer_size),
# #                                                         transforms.CenterCrop(image_size)]),
# #             "to_tensor_transform": transforms.ToTensor(),
# #             "post_tensor_transform": transforms.Normalize(mean, std)
# #         }

#     @classmethod
#     def instantiate_preprocess(
#         cls,
#         train_transform: Dict[str, Union[nn.Module, Callable]] = 'default',
#         val_transform: Dict[str, Union[nn.Module, Callable]] = 'default',
#         test_transform: Dict[str, Union[nn.Module, Callable]] = 'default',
#         predict_transform: Dict[str, Union[nn.Module, Callable]] = 'default',
#         preprocess_cls: Type[Preprocess] = None
#     ) -> Preprocess:
#         """
#         This function is used to instantiate ImageClassificationData preprocess object.
#         Args:
#             train_transform: Train transforms for images.
#             val_transform: Validation transforms for images.
#             test_transform: Test transforms for images.
#             predict_transform: Predict transforms for images.
#             preprocess_cls: User provided preprocess_cls.
#         Example::
#             train_transform = {
#                 "per_sample_transform": T.Compose([
#                     T.RandomResizedCrop(224),
#                     T.RandomHorizontalFlip(),
#                     T.ToTensor(),
#                     T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#                 ]),
#                 "per_batch_transform_on_device": nn.Sequential(K.RandomAffine(360), K.ColorJitter(0.2, 0.3, 0.2, 0.3))
#             }
#         """
#         train_transform, val_transform, test_transform, predict_transform = cls._resolve_transforms(
#             train_transform, val_transform, test_transform, predict_transform
#         )

#         preprocess_cls = preprocess_cls or cls.preprocess_cls
#         preprocess = preprocess_cls(train_transform, val_transform, test_transform, predict_transform)
#         return preprocess

#     @classmethod
#     def _resolve_transforms(
#         cls,
#         train_transform: Optional[Union[str, Dict]] = 'default',
#         val_transform: Optional[Union[str, Dict]] = 'default',
#         test_transform: Optional[Union[str, Dict]] = 'default',
#         predict_transform: Optional[Union[str, Dict]] = 'default',
#     ):

#         if not train_transform or train_transform == 'default':
#             train_transform = cls.default_train_transforms()

#         if not val_transform or val_transform == 'default':
#             val_transform = cls.default_val_transforms()

#         if not test_transform or test_transform == 'default':
#             test_transform = cls.default_val_transforms()

#         if not predict_transform or predict_transform == 'default':
#             predict_transform = cls.default_val_transforms()

#         return (
#             cls._check_transforms(train_transform), cls._check_transforms(val_transform),
#             cls._check_transforms(test_transform), cls._check_transforms(predict_transform)
#         )
    
#     @staticmethod
#     def _check_transforms(transform: Dict[str, Union[nn.Module, Callable]]) -> Dict[str, Union[nn.Module, Callable]]:
#         if transform and not isinstance(transform, Dict):
#             raise MisconfigurationException(
#                 "Transform should be a dict. "
#                 f"Here are the available keys for your transforms: {DataPipeline.PREPROCESS_FUNCS}."
#             )
#         if "per_batch_transform" in transform and "per_sample_transform_on_device" in transform:
#             raise MisconfigurationException(
#                 f'{transform}: `per_batch_transform` and `per_sample_transform_on_device` '
#                 f'are mutual exclusive.'
#             )
#         return transform
    
    

    
    
    
    
    
    
    
if __name__ == "__main__":
    
    seed = 873957
    val_split = 0.2
    batch_size = 16
    name = "PNAS_family_100_512"
    
#     datamodule = PNASImageDataModule(name=name,
#                         batch_size=batch_size,
#                         val_split=val_split,
#                         seed=seed)
    

    
    
    
    
    
    
    
    
    
    
    
    
            
#         self.data.init_datasets(train_transform=transforms.ToTensor(),
#                            eval_transform=transforms.ToTensor())
#         num_classes=len(train_dataset)
#         train_dataloader, val_dataloader, test_dataloader = data.init_dataloaders()
#         # train_dataset.loader(train_dataset.samples[7][0]) #__dir__()
#         PNASImageDataModule.image_size = 224

#         data_module = PNASImageDataModule(train_dataset,
#                                           val_dataset,
#                                           test_dataset,
#                                           batch_size=batch_size,
#                                           num_workers=num_workers)
        
        
#         # download data, train then test
#         MNIST(self.data_dir, train=True, download=True)
#         MNIST(self.data_dir, train=False, download=True)
