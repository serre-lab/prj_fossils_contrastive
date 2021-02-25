import pandas as pd

from data_utils import _clever_crop
import tensorflow as tf


extant_csv_path = '/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/' #extant_family_catalog.csv

train_df = pd.read_csv(extant_csv_path + 'extant_family_10_train.csv')
test_df = pd.read_csv(extant_csv_path + 'extant_family_10_test.csv')

class_labels = sorted(set(data_df[label_col].values))
class_labels_str2int = {label:idx for idx, label in enumerate(class_labels)}
class_labels_int2str = {idx:label for label, idx in class_labels_str2int.items()}

NB_CLASSES = len(class_labels)

def load(path, label):
    x = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    y = class_labels_str2int[label]
    return x, y

def _normalize(x, y):
    x = _clever_crop(x) 
    y = tf.one_hot(y, NB_CLASSES)
    y = tf.cast(y, tf.float32)
    
    x = tf.clip_by_value(x, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
    x = x * 2 - 1

    return x, y

def _remove_label(x, y):
    return x

def _get_dataset(batch_size, supervised, input_col='processed_path', label_col='family'):
    urls_train, labels_train = train_df[input_col], train_df[label_col]
    urls_test, labels_test = test_df[input_col], test_df[label_col]

    ds_train = tf.data.Dataset.from_tensor_slices((urls_train, label_train)).map(load).map(_normalize)
    ds_test  = tf.data.Dataset.from_tensor_slices((urls_test, label_test)).map(load).map(_normalize)

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


if __name__ == '__main__':
    train, test = get_unsupervised(10)
    train, test = get_supervised(10)

    # test to unpack first batch and plot image
    for batch in train.take(1).as_numpy_iterator():
        for x, y in batch:
            plt.imshow(x)
            plt.title(class_labels_int2str[y])
            plt.show()

