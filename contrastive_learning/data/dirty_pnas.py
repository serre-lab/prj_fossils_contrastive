import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Tuple
from contrastive_learning.utils.image_utils import clever_crop


NB_CLASSES = 19
INPUT_SIZE = (128,128)

PNAS_root_dir = "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100"


def _remove_label(x, y):
    return x

def _get_dataset(target_size: Tuple[int]=INPUT_SIZE,
                 batch_size: int=1,
                 grayscale: bool=False,
                 supervised: bool=True,
                 path: str=None):
    path = path or PNAS_root_dir
    ds_train = tfds.folder_dataset.ImageFolder(path).as_dataset(
                                      split='train',
                                      shuffle_files=True,
                                      as_supervised=True)
    ds_test = tfds.folder_dataset.ImageFolder(path).as_dataset(
                                      split='test',
                                      shuffle_files=False,
                                      as_supervised=True)

    _clever_crop = clever_crop(target_size=target_size,
                               grayscale=grayscale)

    def _normalize(x, y):
        x = _clever_crop(x) 
        y = tf.one_hot(y, NB_CLASSES)
        y = tf.cast(y, tf.float32)
        return x, y

    ds_train = ds_train.map(_normalize)
    ds_test = ds_test.map(_normalize)

    if not supervised:
        ds_train = ds_train.map(_remove_label)
        ds_test = ds_test.map(_remove_label)
    
    ds_train = ds_train.batch(batch_size)
    ds_test = ds_test.batch(batch_size)

    return ds_train, ds_test

def get_unsupervised(target_size: Tuple[int]=INPUT_SIZE,
                     batch_size: int=1,
                     grayscale: bool=False,
                     path: str=None):
    return _get_dataset(target_size=target_size,
                        batch_size=batch_size,
                        grayscale=grayscale,
                        supervised=False,
                        path=path)

def get_supervised(target_size: Tuple[int]=INPUT_SIZE,
                   batch_size: int=1,
                   grayscale: bool=False,
                   path: str=None):
    return _get_dataset(target_size=target_size,
                        batch_size=batch_size,
                        grayscale=grayscale,
                        supervised=True,
                        path=path)
