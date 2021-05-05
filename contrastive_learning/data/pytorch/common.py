"""
contrastive_learning.data.pytorch.common


Common classes & functions for simplifying the boilerplate code in definitions of custom datasets in this repo.

Created by: Sunday April 25th, 2021
Author: Jacob A Rose





"""
from torchvision.datasets import folder, vision, ImageFolder
from torch.utils.data import Dataset, Subset, random_split
from typing import Sequence
from typing import Any, Callable, List, Optional, Tuple
import random
import numpy as np
import torch
from pathlib import Path

__all__ = ['LeavesDataset', 'seed_worker', 'TrainValSplitDataset', 'SubsetImageDataset']



class LeavesDataset(ImageFolder):

    splits_on_disk : Tuple[str]= ("train", "test")
    
    def __init__(
            self,
            name: str=None,
            split: str="train",
            **kwargs: Any
            ) -> None:

        assert split in self.splits_on_disk
        assert name in self.available_datasets, f"{name} is not in the set of available datasets. Please try one of the following: \n{self.available_datasets.keys()}"
        
        self.name = name
        self.split = split
        self.dataset_dir = Path(self.available_datasets[name])
        self.split_dir = self.dataset_dir / self.split
        
        super().__init__(root=self.split_dir,
                         **kwargs)

    @property
    def available_datasets(self):
        """
        Subclasses must define this property
        Must return a dict mapping dataset key names to their absolute paths on disk.
        """
        raise NotImplementedError
        
    def __repr__(self):
        content = super().__repr__()
        content += f'\n    Name: {self.name}'
        content += f'\n    Split: {self.split}'
        return content



def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
    
    
class TrainValSplitDataset(ImageFolder):
    """
    Use this class as a Factory for creating train and val splits from a single dataset, 
    returning 2 separate datasets with no shared references (as opposed to the native SubsetDataset)
    
    Example
    =======
    >> train_dataset = ImageFolder(root=train_dir)
    >> train_data, val_data = TrainValSplitDataset.train_val_split(train_dataset, val_split=0.2, seed=0)
    
    or, identically,
    
    >> train_dataset = TrainValSplitDataset(root=train_dir)
    >> train_data, val_data = TrainValSplitDataset.train_val_split(train_dataset, val_split=0.2, seed=0)
    """
    
    all_params: List[str]= [
                            'class_to_idx',
                            'classes',
                            'extensions',
                            'imgs',
                            'loader',
                            'root',
                            'samples',
                            'target_transform',
                            'targets',
                            'transform',
                            'transforms'
                            ]
    sample_params: List[str] = ['imgs',
                                'samples',
                                'targets']
    
    @classmethod
    def train_val_split(cls, full_dataset, val_split: float=0.2, seed: float=None) -> Tuple[ImageFolder]:
        
        num_samples = len(full_dataset)
        split_idx = (int(np.floor((1-val_split) * num_samples)),
                     int(np.floor(val_split * num_samples)))
        if seed is None:
            generator = None
        else:
            generator = Generator().manual_seed(seed)

        train_data, val_data = random_split(full_dataset, 
                                            split_idx,
                                            generator=generator)
        
        train_dataset = cls.select_from_dataset(full_dataset, indices=train_data.indices)
        val_dataset = cls.select_from_dataset(full_dataset, indices=val_data.indices)
        
        return train_dataset, val_dataset
        
    @classmethod
    def from_dataset(cls, dataset):
        new_dataset = cls(root=dataset.root)
        
        for key in cls.all_params:
            if hasattr(dataset, key):
                setattr(new_dataset, key, getattr(dataset, key))
                         
        return new_dataset
    
    
    @classmethod
    def select_from_dataset(cls, dataset, indices=None):
        upgraded_dataset = cls.from_dataset(dataset)
        return upgraded_dataset.select(indices)
    
    
    def select(self, indices):
        new_subset = self.from_dataset(self)
        for key in self.sample_params:
            old_attr = getattr(self, key)
            new_attr = [old_attr[idx] for idx in indices]
            setattr(new_subset, key, new_attr)
        return new_subset

                
    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = ["Number of datapoints: {}".format(self.__len__())]
        if self.root is not None:
            body.append("Root location: {}".format(self.root))
        body += self.extra_repr().splitlines()
        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return '\n'.join(lines)

    def _format_transform_repr(self, transform: Callable, head: str) -> List[str]:
        lines = transform.__repr__().splitlines()
        return (["{}{}".format(head, lines[0])] +
                ["{}{}".format(" " * len(head), line) for line in lines[1:]])





    
    
    
###############################
###############################
# class SubsetImageDataset(folder.ImageFolder):
class SubsetImageDataset(folder.DatasetFolder):
# class SubsetDataset(Dataset):
    """
    Custom class for creating a Subset of a Dataset while retaining the built-in methods/attributes/properties of Datasets.
    
    User provides a full dataset to be split, along with indices for inclusion in this subset.

    Arguments:
        data (Dataset): The whole Dataset
        indices (sequence): Indices in the whole set selected for subset
    """
    def __init__(self,
                 data: Dataset,
                 indices: Sequence,
                 transforms: Optional[Callable] = None,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 ):
        subset = Subset(data, indices)
        self.indices = subset.indices
        data = subset.dataset

        self.class_to_idx = data.class_to_idx
        self.classes = data.classes
        self.root = data.root
        self.loader = data.loader
        self.extensions = data.extensions
                
        self.samples = [data.samples[idx] for idx in self.indices]
        self.targets = [data.targets[idx] for idx in self.indices]

        
        #######################################
        transforms = transforms or data.transforms
        transform = transform or data.transform
        target_transform = target_transform or data.target_transform
        
        has_separate_transform = transform is not None or target_transform is not None

        self.transform = transform or data.transform
        self.target_transform = target_transform or data.target_transform
        if has_separate_transform:
            transforms = vision.StandardTransform(transform, target_transform)
        self.transforms = transforms
        

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         image, target = self.samples[idx]
#         return image, target

    def __len__(self):
        return len(self.targets)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
#######################################
#######################################


    
    


# class SubsetImageDataset(folder.ImageFolder):
#     """
#     Custom class for creating a Subset of a Dataset while retaining the built-in methods/attributes/properties of Datasets.
    
#     User provides a full dataset to be split, along with indices for inclusion in this subset.

#     Arguments:
#         dataset (Dataset): The whole Dataset
#         indices (sequence): Indices in the whole set selected for subset
#     """
#     def __init__(self,
# #         super().__init__(
# #                          root: str,
# #                          transform: Optional[Callable] = None,
# #                          target_transform: Optional[Callable] = None,
# #                          loader: Callable[[str], Any] = default_loader,
# #                          is_valid_file: Optional[Callable[[str], bool]] = None,
# #                         ):
#                  dataset: Dataset,
#                  indices: Sequence
#                 ):
#         super(DatasetFolder, self).__init__(root, transform=transform,
#                                             target_transform=target_transform)
#         classes, class_to_idx = self._find_classes(self.root)
#         samples = self.make_dataset(self.root, class_to_idx, extensions, is_valid_file)

#         self.loader = loader
#         self.extensions = extensions

#         self.classes = classes
#         self.class_to_idx = class_to_idx
#         self.samples = samples
#         self.targets = [s[1] for s in samples]

        
        
# #         self.dataset = Subset(dataset, indices)
#         self.dataset = dataset[indices]
#         self.samples = self.dataset.samples
#         self.targets = self.dataset.targets

#     def __getitem__(self, idx):
#         if torch.is_tensor(idx):
#             idx = idx.tolist()
#         image, target = self.samples[idx]
#         return image, target

#     def __len__(self):
#         return len(self.targets)
