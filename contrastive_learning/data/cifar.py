import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def _normalize(x, y):
    return x.astype('float32'), tf.one_hot(y[:,0], 10).numpy()


def get_unsupervised(batch_size=128, val_split=0.2):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train, y_train = _normalize(x_train, y_train)
    x_test, y_test = _normalize(x_test, y_test)

    num_samples = len(x_train)
    train_samples = int((1-val_split)*num_samples)

    x_train, x_val = x_train[:train_samples], x_train[train_samples:]

    train_dataset = tf.data.Dataset.from_tensor_slices(x_train)
    val_dataset = tf.data.Dataset.from_tensor_slices(x_val)
    test_dataset = tf.data.Dataset.from_tensor_slices(x_test)

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset


def get_supervised(batch_size=128, val_split=0.2):
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train, y_train = _normalize(x_train, y_train)
    x_test, y_test = _normalize(x_test, y_test)

    num_samples = len(x_train)
    train_samples = int((1-val_split)*num_samples)

    x_train, x_val = x_train[:train_samples], x_train[train_samples:]
    y_train, y_val = y_train[:train_samples], y_train[train_samples:]

    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))

    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
    val_dataset = val_dataset.batch(batch_size)
    test_dataset = test_dataset.batch(batch_size)

    return train_dataset, val_dataset, test_dataset