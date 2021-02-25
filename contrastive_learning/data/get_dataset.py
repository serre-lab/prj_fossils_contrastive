import tensorflow as tf

#from contrastive_learning.transformations import (pad, random_blur, random_distorsion, 
#                               random_flip, random_jitter, random_scale,
#                               compose_transformations)
import contrastive_learning.data.cifar as cifar
import contrastive_learning.data.dirty_pnas as pnas

def augmentations(x, crop_size=22, brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2):
    x = tf.cast(x, tf.float32)
    x = tf.image.random_crop(x, (tf.shape(x)[0], 100, 100, 3))
    x = tf.image.random_brightness(x, max_delta=brightness)
    x = tf.image.random_contrast(x, lower=1.0-contrast, upper=1+contrast)
    x = tf.image.random_saturation(x, lower=1.0-saturation, upper=1.0+saturation)
    x = tf.image.random_hue(x, max_delta=hue)
    x = tf.image.resize(x, (128, 128))
    x = tf.clip_by_value(x, 0.0, 255.0)
    x = tf.keras.applications.resnet_v2.preprocess_input(x)
    return x


def symmetric_batch(batch_x):
    return tf.concat([augmentations(batch_x), augmentations(batch_x)], axis=0)

def resize_images(batch_x, width=224, height=224):
    return tf.image.resize(batch_x, (width, height))

def get_dataset(dataset='cifar_unsup', batch_size=128, val_split=0.2,path='dataset'):
    # base dataset
    if dataset == 'cifar_unsup':
        train, val, test = cifar.get_unsupervised(batch_size=batch_size//2, val_split=val_split)
        # apply symmetric batch mechanism
        train = train.map(symmetric_batch)
        test = test.map(symmetric_batch)
        val = val.map(symmetric_batch)
    elif dataset == 'cifar_sup':
        train, val, test = cifar.get_supervised(batch_size=batch_size//2, val_split=val_split)
        train = train.map(lambda x,y : (resize_images(x), y) )  
        val = val.map(lambda x,y : (resize_images(x), y))
        test = test.map(lambda x,y : (resize_images(x), y))
    elif dataset == 'pnas_unsup':
        train, test = pnas.get_unsupervised(batch_size=batch_size//2, path=path)
        train = train.map(symmetric_batch) 
        test = test.map(symmetric_batch)
        val = test
    elif dataset == 'pnas_sup':
        train, test = pnas.get_supervised(batch_size=batch_size//2, path=path)
        val = test
    else: 
        raise NotImplementedError('Dataset is not implemented yet')

    return train, val, test
