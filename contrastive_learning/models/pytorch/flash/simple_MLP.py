"""
contrastive_learning.models.pytorch.flash.simple_MLP

Simple Demo model for isolation and debugging purposes

Created by: Wednesday April 14th, 2021
Author: Jacob A Rose



Multi-Layer Perceptron (MLP) custom implementation defined by subclassing a PyTorch-Lightning LightningModule and overriding a subset of its methods.

* By default, the MLP is configured to be a 2-layer perceptron, with two fully-connected layers followed by a fully-connected softmax readout layer.


## Building a Model with Lightning
In PyTorch Lightning, models are built with LightningModule (docs here), which has all the functionality of a vanilla torch.nn.Module (🍦) but with a few delicious cherries of added functionality on top (🍨). These cherries are there to cut down on boilerplate and help separate out the ML engineering code from the actual machine learning.

We'll demonstrate this process with LitMLP, which applies a two-layer perceptron (aka two fully-connected layers and a fully-connected softmax readout layer) to input Tensors.

Note: It is common in the Lightning community to shorten "Lightning" to "Lit". This sometimes it sound like your code was written by Travis Scott. We consider this a good thing.



Custom LitMLP overriden methods:


training_step: which takes a batch and computes the loss; backprop goes through it
configure_optimizers: which returns the torch.optim.Optimizer to apply after the training_step


"""

import numpy as np

import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
import wandb


class LitMLP(pl.LightningModule):

    def __init__(self, in_dims, n_classes=10,
                 n_layer_1=128, n_layer_2=256, lr=1e-4):
        super().__init__()

        # we flatten the input Tensors and pass them through an MLP
        self.layer_1 = nn.Linear(np.prod(in_dims), n_layer_1)
        self.layer_2 = nn.Linear(n_layer_1, n_layer_2)
        self.layer_3 = nn.Linear(n_layer_2, n_classes)

        # log hyperparameters
        self.save_hyperparameters()

        # compute the accuracy -- no need to roll your own!
        self.train_acc = pl.metrics.Accuracy()
        self.valid_acc = pl.metrics.Accuracy()
        self.test_acc = pl.metrics.Accuracy()

    def forward(self, x):
        """
        Defines a forward pass using the Stem-Learner-Task
        design pattern from Deep Learning Design Patterns:
        https://www.manning.com/books/deep-learning-design-patterns
        """
        batch_size, *dims = x.size()

        # stem: flatten
        x = x.view(batch_size, -1)

        # learner: two fully-connected layers
        x = F.relu(self.layer_1(x))
        x = F.relu(self.layer_2(x))
        
        # task: compute class logits
        x = self.layer_3(x)
        x = F.log_softmax(x, dim=1)

        return x

    # convenient method to get the loss on a batch
    def loss(self, xs, ys):
        logits = self(xs)  # this calls self.forward
        loss = F.nll_loss(logits, ys)
        return logits, loss
    
    
    def training_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        # logging metrics we calculated by hand
        self.log('train/loss', loss, on_epoch=True)
        # logging a pl.Metric
        self.train_acc(preds, ys)
        self.log('train/acc', self.train_acc, on_epoch=True)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])
    
    
    def test_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)

        self.test_acc(preds, ys)
        self.log("test/loss_epoch", loss, on_step=False, on_epoch=True)
        self.log("test/acc_epoch", self.test_acc, on_step=False, on_epoch=True)


    def test_epoch_end(self, test_step_outputs):  # args are defined as part of pl API
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = "model_final.onnx"
        torch.onnx.export(self, dummy_input, model_filename)
        wandb.save(model_filename)
        
    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        logits, loss = self.loss(xs, ys)
        preds = torch.argmax(logits, 1)
        self.valid_acc(preds, ys)

        self.log("valid/loss_epoch", loss)  # default on val/test is on_epoch only
        self.log('valid/acc_epoch', self.valid_acc)

        return logits

    def validation_epoch_end(self, validation_step_outputs):
        dummy_input = torch.zeros(self.hparams["in_dims"], device=self.device)
        model_filename = f"model_{str(self.global_step).zfill(5)}.onnx"
        torch.onnx.export(self, dummy_input, model_filename)
        wandb.save(model_filename)

        flattened_logits = torch.flatten(torch.cat(validation_step_outputs))
        self.logger.experiment.log(
            {"valid/logits": wandb.Histogram(flattened_logits.to("cpu")),
             "global_step": self.global_step})

        
        
        
if __name__ == "__main__":
    
    from contrastive_learning.data.pytorch.mnist import MNISTDataModule
    from contrastive_learning.data.pytorch.flash.logging import ImagePredictionLogger
    from pytorch_lightning.loggers import WandbLogger

    mnist = MNISTDataModule()
    mnist.prepare_data()
    mnist.setup()

    # grab samples to log predictions on
    samples = next(iter(mnist.val_dataloader()))
    
#     wandb_logger = WandbLogger(project="lit-wandb")
    wandb_logger = WandbLogger(project="code-test-results", group="simple_MLP.py", settings=wandb.Settings(start_method="fork"))
    trainer = pl.Trainer(
                         logger=wandb_logger,    # W&B integration
                         log_every_n_steps=50,   # set the logging frequency
                         gpus=1,                # use all GPUs
                         max_epochs=5,           # number of epochs
                         deterministic=True,     # keep it deterministic
                         callbacks=[ImagePredictionLogger(samples)]
                         )
    
    
    # setup model
    model = LitMLP(in_dims=(1, 28, 28))

    # fit the model
    trainer.fit(model, mnist)

    # evaluate the model on a test set
    trainer.test(datamodule=mnist,
                 ckpt_path=None)  # uses last-saved model

    wandb.finish()