import tensorflow as tf
import tensorflow_datasets as tfds

NB_CLASSES = 19
INPUT_SIZE = 128

def _clever_crop(x):
    return tf.image.resize(x, INPUT_SIZE) #Todo: Ivan & Jacob we need your clever crop

def _normalize(x, y):
  x = _clever_crop(x) #Todo: Ivan & Jacob we need your clever crop
  y = tf.one_hot(y, NB_CLASSES)
  y = tf.cast(y, tf.float32)
  return x, y

def _remove_label(git x, y):
    return x

def _get_dataset(batch_size, supervised, path):
    ds_train = tfds.folder_dataset.ImageFolder(path).as_dataset(split='train', shuffle_files=True, as_supervised=True)
    ds_test = tfds.folder_dataset.ImageFolder(path).as_dataset(split='test', shuffle_files=True, as_supervised=True)

    ds_train = ds_train.map(_normalize)
    ds_test = ds_test.map(_normalize)

    if not supervised:
        ds_train = ds_train.map(_remove_label)
        ds_test = ds_test.map(_remove_label)
    
    ds_train = ds_train.batch(batch_size)
    ds_test = ds_test.batch(batch_size)

    return ds_train, ds_test

def get_unsupervised(batch_size, path="dataset"):
    return _get_dataset(batch_size, supervised=False, path=path)

def get_supervised(batch_size, path="dataset"):
    return _get_dataset(batch_size, supervised=True, path=path)