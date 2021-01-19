import tensorflow as tf

from ..transformations import (pad, random_blur, random_distorsion, 
                               random_flip, random_jitter, random_scale,
                               compose_transformations)
from .cifar import get_data
import os 

def  resize(size):
    def transform(images):
        return tf.image.resize(images, (size, size))
    return transform

augmentation_function = compose_transformations([
    pad(24, 0.0),
    random_jitter(6),
    random_blur(sigma_range=[0.8, 2.0]),
    random_jitter(12),
    random_scale([0.95, 0.99]),
    random_jitter(12),
    random_distorsion(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    resize(224)
    # TODO : clip to ensure [0, 255] ?
])

def symmetric_batch(batch_x):
    #sseed = int.from_bytes(os.urandom(4), byteorder='little')
    #tf.random.seed(sseed)
    return tf.concat([augmentation_function(batch_x), augmentation_function(batch_x)], axis=0)

train_dataset, val_dataset, test_dataset = get_data(batch_size=128, val_split=0.2)

train_dataset = train_dataset.map(symmetric_batch)
val_dataset = train_dataset.map(val_dataset)
test_dataset = train_dataset.map(test_dataset)


