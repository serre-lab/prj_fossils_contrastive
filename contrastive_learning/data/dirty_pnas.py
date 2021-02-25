import tensorflow as tf
import tensorflow_datasets as tfds

NB_CLASSES = 19
INPUT_SIZE = (128,128)

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
    
    return img #Todo: Ivan & Jacob we need your clever crop

def _normalize(x, y):
  x = _clever_crop(x) #Todo: Ivan & Jacob we need your clever crop
  y = tf.one_hot(y, NB_CLASSES)
  y = tf.cast(y, tf.float32)
  return x, y

def _remove_label(x, y):
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