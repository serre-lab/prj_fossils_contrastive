
import tensorflow as tf
from tensorflow.keras.datasets import cifar10

from contrastive_learning.utils.label_utils import ClassLabelEncoder

def _normalize(x, y):
    return x.astype('float32'), tf.one_hot(y[:,0], 10).numpy()

def _normalize_and_resize(x, y,width=224,height=224):
    return tf.image.resize(x.astype('float32'),(width,height)), tf.one_hot(y[:,0], 10).numpy()


def load_and_extract_cifar10(batch_size=128, val_split=0.2):

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train, y_train = _normalize(x_train, y_train)
    x_test, y_test = _normalize(x_test, y_test)

    num_samples = len(x_train)
    train_samples = int((1-val_split)*num_samples)

    x_train, x_val = x_train[:train_samples], x_train[train_samples:]
    y_train, y_val = y_train[:train_samples], y_train[train_samples:]

    return {'train':(x_train, y_train),
            'val':(x_val, y_val),
            'test':(x_test, y_test)}


def get_unsupervised(batch_size=128, val_split=0.2):

    data = load_and_extract_cifar10(batch_size=batch_size, val_split=val_split)

    x_train, _ = data['train']
    x_val, _ = data['val']
    x_test, _ = data['test']

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset


def get_supervised(batch_size=128, val_split=0.2, seed: int=None, return_label_encoder: bool=False):

    data = load_and_extract_cifar10(batch_size=batch_size, val_split=val_split)

    x_train, y_train = data['train']
    x_val, y_val = data['val']
    x_test, y_test = data['test']

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    if return_label_encoder:
        label_encoder = ClassLabelEncoder(y_train, name='cifar10', alphabetical=True)

        return (train_dataset, val_dataset, test_dataset), label_encoder

    return train_dataset, val_dataset, test_dataset


# def get_unsupervised(batch_size=128, val_split=0.2):
#     (x_train, y_train), (x_test, y_test) = cifar10.load_data()

#     x_train, y_train = _normalize(x_train, y_train)
#     x_test, y_test = _normalize(x_test, y_test)

#     num_samples = len(x_train)
#     train_samples = int((1-val_split)*num_samples)

#     x_train, x_val = x_train[:train_samples], x_train[train_samples:]

#     train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
#     val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
#     test_dataset = tf.data.Dataset.from_tensor_slices(x_test)

#     train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
#     val_dataset = val_dataset.batch(batch_size)
#     test_dataset = test_dataset.batch(batch_size)

#     return train_dataset, val_dataset, test_dataset


# def get_supervised(batch_size=128, val_split=0.2):
#     (x_train, y_train), (x_test, y_test) = cifar10.load_data()

#     x_train, y_train = _normalize(x_train, y_train)
#     x_test, y_test = _normalize(x_test, y_test)

#     num_samples = len(x_train)
#     train_samples = int((1-val_split)*num_samples)

#     x_train, x_val = x_train[:train_samples], x_train[train_samples:]
#     y_train, y_val = y_train[:train_samples], y_train[train_samples:]


#     train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
#     val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
#     test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

#     train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
#     val_dataset = val_dataset.batch(batch_size)
#     test_dataset = test_dataset.batch(batch_size)

#     return train_dataset, val_dataset, test_dataset