

from torchvision.datasets import folder, vision
from torch.utils.data import Dataset, Subset
from typing import Sequence
from typing import Any, Callable, List, Optional, Tuple
import random
import numpy as np
import torch


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


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