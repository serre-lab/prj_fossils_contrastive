import tensorflow as tf
from tensorflow.keras.datasets import cifar10

def _normalize(x, y):
    return x.astype('float32'), tf.one_hot(y[:,0], 10).numpy()

def _normalize_and_resize(x, y,width=224,height=224):
    return tf.image.resize(x.astype('float32'),(width,height)), tf.one_hot(y[:,0], 10).numpy()

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

def _clever_crop(img):
    maxside = tf.math.maximum(tf.shape(img)[0],tf.shape(img)[1])
    minside = tf.math.minimum(tf.shape(img)[0],tf.shape(img)[1])
    new_img = img
             
    if tf.math.divide(maxside,minside) > 1.2:
        repeating = tf.math.floor(tf.math.divide(maxside,minside))  
        new_img = img
        if tf.math.equal(tf.shape(img)[1],minside):
            for i in range(int(repeating)):
                new_img = tf.concat((new_img, img), axis=1) 

        if tf.math.equal(tf.shape(img)[0],minside):
            for i in range(int(repeating)):
                new_img = tf.concat((new_img, img), axis=0)
            new_img = tf.image.rot90(new_img) 
    else:
        new_img = img      
    img = tf.image.resize(new_img, INPUT_SIZE)
    if grayscale:
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.grayscale_to_rgb(img)
        #img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
        # 
    img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
    img = img * 2 - 1
    
    return img 