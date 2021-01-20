import tensorflow as tf

from contrastive_learning.transformations import (pad, random_blur, random_distorsion, 
                               random_flip, random_jitter, random_scale,
                               compose_transformations)
import contrastive_learning.data.cifar as cifar

def  resize(size):
    def transform(images):
        return tf.image.resize(images, (size, size))
    return transform

augmentation_function = compose_transformations([
    pad(24, 0.0),
    random_jitter(2),
    random_blur(sigma_range=[0.8, 2.0]),
    #random_jitter(12),
    random_scale([0.95, 0.99]),
    resize(224),
    #random_jitter(12),
    #random_distorsion(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
    # TODO : clip to ensure [0, 255] ?
])

def symmetric_batch(batch_x):
    batch_x = tf.concat([batch_x, batch_x], axis=0)
    return augmentation_function(batch_x)

def get_dataset(dataset='cifar', batch_size=128, val_split=0.2):
    # base dataset
    if dataset == 'cifar':
        train, val, test = cifar.get_unsupervised(batch_size=batch_size//2, val_split=val_split)
    elif dataset == 'leaves':
        raise NotImplementedError('Leaves dataset is not implemented yet')
    else: 
        raise NotImplementedError('Dataset is not implemented yet')
    
    # apply symmetric batch mechanism
    train = train.map(symmetric_batch)
    test = test.map(symmetric_batch)
    val = val.map(symmetric_batch)

    return train, val, test


