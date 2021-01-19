import tensorflow as tf

from contrastive_learning.data.get_dataset import get_dataset
from contrastive_learning.losses import get_contrastive_loss
from contrastive_learning.train_contrastive import train_contrastive
from contrastive_learning.models.resnet import ResNetSimCLR


if __name__ == "__main__":
    loss = get_contrastive_loss(temperature=1.0)
    batch_size = 32
    train, val, test = get_dataset('cifar', batch_size=batch_size)
    input_shape = (224,224,3)
    out_dim = batch_size*2
    projector = ResNetSimCLR(input_shape,out_dim )

    train_contrastive(train, val, test,projector, loss, tf.keras.optimizers.Adam(), epochs=100, verbose=True, froze_backbone=True)
