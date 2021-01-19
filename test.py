import tensorflow as tf

from data.get_dataset import get_dataset
from losses import get_contrastive_loss
from train_contrastive import train_contrastive

loss = get_contrastive_loss(temperature=1.0)
train, val, test = get_dataset('cifar', batch_size=32)

train_contrastive(train, val, test, loss, tf.keras.optimizers.Adam(), epochs=100, verbose=True, froze_backbone=True)