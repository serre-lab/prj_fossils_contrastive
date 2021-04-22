


num_workers = 1

backbone = "resnet50"
learning_rate = 1e-4
batch_size = 2
num_epochs = 30
# feature_extract=True
seed = 938462




import random
import torch
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.is_available()
# head = None


from torchmetrics import Accuracy
from flash import Task
from torch import nn, optim
import flash
from flash.vision import ImageClassificationData, ImageClassifier
from torchvision import models
import pytorch_lightning as pl
from typing import List, Callable, Dict, Union, Type, Optional
from pytorch_lightning.callbacks import Callback
from contrastive_learning.data.pytorch.flash.process import Preprocess, Postprocess
from contrastive_learning.data.pytorch.pnas import PNASImageDataModule

from torchvision import transforms
from contrastive_learning.data.pytorch import pnas



if __name__ == "__main__":

    data = pnas.PNASLeaves()
    train_dataset, val_dataset, test_dataset = data.init_datasets(train_transform=transforms.ToTensor(),
                                                                  eval_transform=transforms.ToTensor())
    num_classes=len(train_dataset)
    train_dataloader, val_dataloader, test_dataloader = data.init_dataloaders()
    # train_dataset.loader(train_dataset.samples[7][0]) #__dir__()
    PNASImageDataModule.image_size = 224

    data_module = PNASImageDataModule(train_dataset,
                                      val_dataset,
                                      test_dataset,
                                      batch_size=batch_size,
                                      num_workers=num_workers)

    classifier = ImageClassifier(num_classes=num_classes,
                                 backbone = backbone,
                                 pretrained = True,
                                 loss_fn = torch.nn.functional.cross_entropy,
                                 optimizer = torch.optim.Adam,
                                 metrics = Accuracy(),
                                 learning_rate=learning_rate)    

    trainer = flash.Trainer(max_epochs=num_epochs, gpus=1, log_gpu_memory='all')#True)
    # trainer.fit(classifier, train_dataloader, val_dataloader) # datamodule=data_module)
    trainer.finetune(classifier, train_dataloader, val_dataloader, strategy='freeze')