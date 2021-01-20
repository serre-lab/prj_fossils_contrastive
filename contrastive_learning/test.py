import tensorflow as tf
import matplotlib 
matplotlib.use('Agg')
from contrastive_learning.data.get_dataset import get_dataset
from contrastive_learning.losses import get_contrastive_loss
from contrastive_learning.train_contrastive import train_contrastive
from contrastive_learning.models.resnet import ResNetSimCLR


if __name__ == "__main__":
    loss = get_contrastive_loss(temperature=1.0)
    batch_size = 32
    train, val, test = get_dataset('cifar_unsup', batch_size=batch_size)
    input_shape = (224,224,3)
    out_dim = batch_size*2
    projector = ResNetSimCLR(input_shape,out_dim )

    import numpy as np
    import matplotlib.pyplot as plt
    temp = next(iter(train))
    for i in range(10):
        plt.subplot(1, 10, i+1)
        plt.imshow(temp[i])
    plt.show()

    for i in range(10):
        plt.subplot(1, 2, 1)
        plt.imshow(temp[i])
        plt.subplot(1, 2, 2)
        plt.imshow(temp[i+batch_size//2])
        plt.savefig('test.png')
        plt.show()
        

    train_contrastive(train, val, test,projector, loss, tf.keras.optimizers.Adam(), epochs=100, verbose=True, froze_backbone=True)
