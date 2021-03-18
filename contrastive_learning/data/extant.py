# import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib 
matplotlib.use('agg')
import matplotlib.pyplot as plt
from functools import partial
from sklearn.model_selection import train_test_split
# from contrastive_learning.utils.data_utils import DatasetInfo
from contrastive_learning.data import stateful
from contrastive_learning.utils.image_utils import _clever_crop


class ExtantDatasetInfo(stateful.DatasetInfo):

    def __init__(self):
        super().__init__(dataset_name='extant',
                         label_col='family',
                         threshold=4)


def load(path, label):
    x = tf.image.decode_jpeg(tf.io.read_file(path), channels=3)
    return x, label

def _normalize(x, y, size, num_classes):
    x = _clever_crop(x, (size, size)) 
    y = tf.one_hot(y, num_classes)
    y = tf.cast(y, tf.float32)
    
    x = tf.clip_by_value(x, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
    x = x * 2 - 1

    return x, y

def _remove_label(x, y):
    return x

def _get_dataset(batch_size, 
                 size,
                 supervised=True,
                 input_col='processed_path',
                 label_col='label',
                 seed: int=None):
    ds_info = ExtantDatasetInfo()

    train_df, val_df, test_df = ds_info.as_dataframes()

    x_train, y_train = train_df[input_col], train_df[label_col]
    x_val, y_val = val_df[input_col], val_df[label_col]
    x_test, y_test = test_df[input_col], test_df[label_col]
    
    normalize = partial(_normalize, size=size, num_classes=ds_info.NUM_CLASSES)
    
    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(load, num_parallel_calls=-1).map(normalize, num_parallel_calls=-1)
    ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val)).map(load, num_parallel_calls=-1).map(normalize, num_parallel_calls=-1)
    ds_test  = tf.data.Dataset.from_tensor_slices((x_test, y_test)).map(load, num_parallel_calls=-1).map(normalize, num_parallel_calls=-1)

    if not supervised:
        ds_train = ds_train.map(_remove_label)
        ds_val = ds_val.map(_remove_label)
        ds_test = ds_test.map(_remove_label)
    
    ds_train = ds_train.batch(batch_size)
    ds_val = ds_val.batch(batch_size)
    ds_test = ds_test.batch(batch_size)

    return ds_train, ds_val, ds_test

#####################################################################

def get_unsupervised(batch_size, size, seed: int=None):
    return _get_dataset(batch_size,size, supervised=False, seed=seed)

def get_supervised(batch_size, size, seed: int=None):
    return _get_dataset(batch_size, size, supervised=True, seed=seed)

#####################################################################

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def create_tf_feature(img, label):
    features = {
        'raw': _bytes_feature(img),
        'label': _int64_feature(label)
    }
    return tf.train.Example(features=tf.train.Features(feature=features))



def decode_example(example, num_classes: int, target_size):
    feature_description = {
                            'raw': tf.io.FixedLenFeature([], tf.string),
                            'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
                            }
    features = tf.io.parse_single_example(example,features=feature_description)

    img = tf.image.decode_jpeg(features['raw'], channels=3) # * 255.0
    img = tf.image.resize(img, target_size)

    label = tf.cast(features['label'], tf.int32)
    label = tf.one_hot(label, depth=num_classes)

    return img, label






if __name__ == '__main__':
    train, test = get_unsupervised(256,256)
    train, test = get_supervised(256,256)
    from time import time 
    # test to unpack first batch and plot image
    tic_inner = tic = time()
    shard_size = 1024
    data_df = train.unbatch().batch(shard_size)

    for i, shard_i in enumerate(data_df):
        toc_inner = time()
        print(f'{toc_inner-tic_inner:.2f} s')
        tic_inner = time()
        record_file = f'/media/data_cifs/projects/prj_fossils/data/processed_data/tf_records_2021_v1/images_{i}.tfrecords'
        with tf.io.TFRecordWriter(record_file) as writer:

            num_samples_in_shard = shard_i[0].shape[0]

            for j in range(num_samples_in_shard):
                img, label = shard_i[0][i,...].numpy(), shard_i[1][i,...].numpy()
                label = tf.argmax(label)
                # cv2.imwrite('temp/img.jpg', img)
                # img_string = open('temp/img.jpg', 'rb').read()

                img = tf.image.encode_jpeg(img, optimize_size=True, chroma_downsampling=False)
                tf_example = create_tf_feature(img, label)
                writer.write(tf_example.SerializeToString())   
        # batch_i += 1

        # if batch_i > 20:
        #     shard_i += 1
        #     batch_i = 0

    toc = time()
    print(toc-tic)

    # for batch in iter(train):
    #     print(batch[0].shape,batch[1].shape)
    #     toc_inner = time()
    #     print(f'{toc_inner-tic_inner:.2f} s')
    #     tic_inner = time()

    #     record_file = f'/media/data_cifs/projects/prj_fossils/data/processed_data/tf_records_2021_v1/images_{shard_i}.tfrecords'
    #     with tf.io.TFRecordWriter(record_file) as writer:
    #         for img, label in zip(batch[0], batch[1]):
    #             img = np.array(img)
    #             label = tf.argmax(label)
    #             # cv2.imwrite('temp/img.jpg', img)
    #             # img_string = open('temp/img.jpg', 'rb').read()

    #             img = tf.image.encode_jpeg(img, optimize_size=True, chroma_downsampling=False)
    #             tf_example = create_tf_feature(img, label)
    #             writer.write(tf_example.SerializeToString())        
    #     batch_i += 1

    #     if batch_i > 20:
    #         shard_i += 1
    #         batch_i = 0

    # toc = time()
    # print(toc-tic)
    
##################################
    # def parse_image(self, src_filepath, label):

    #     img = tf.io.read_file(src_filepath)
    #     img = tf.image.decode_image(img, channels=3)
    #     img = tf.compat.v1.image.resize_image_with_pad(img, *self.target_size)
    #     return img, label
        
    # def encode_example(self, img, label):
    #     img = tf.image.encode_jpeg(img, optimize_size=True, chroma_downsampling=False)

    #     features = {
    #                 'image/bytes': bytes_feature(img),
    #                 'label': int64_feature(label)
    #                 }
    #     example_proto = tf.train.Example(features=tf.train.Features(feature=features))
    #     return example_proto.SerializeToString()
    # def decode_example(self, example):
    #     feature_description = {
    #                             'image/bytes': tf.io.FixedLenFeature([], tf.string),
    #                             'label': tf.io.FixedLenFeature([], tf.int64, default_value=-1)
    #                             }
    #     features = tf.io.parse_single_example(example,features=feature_description)

    #     img = tf.image.decode_jpeg(features['image/bytes'], channels=3) # * 255.0
    #     img = tf.compat.v1.image.resize_image_with_pad(img, *self.target_size)

    #     label = tf.cast(features['label'], tf.int32)
    #     label = tf.one_hot(label, depth=self.num_classes)

    #     return img, label
###############################


    # def _bytes_feature(value):
    # """Returns a bytes_list from a string / byte."""
    #     if isinstance(value, type(tf.constant(0))):
    #         value = value.numpy() # BytesList won't unpack a string from an EagerTensor.
    #     return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    # def _float_feature(value):
    #     """Returns a float_list from a float / double."""
    #     return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

    # def _int64_feature(value):
    #     """Returns an int64_list from a bool / enum / int / uint."""
    #     return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

    # def create_tf_feature(img, label):
    #     features = {
    #         'raw': _bytes_feature(img),
    #         'label': _int64_feature(label)
    #     }
    #     return tf.train.Example(features=tf.train.Features(feature=features))


    # first pass, construct random filename for each shard
    # filenames_for_shard = [ [] for _ in range(nb_models) ]

    # for shard_i in range(nb_model):
    # for classname in classes_files.keys():
    #     class_id = class_to_id[classname]
    #     pack_size = pack_per_class[class_id]
    #     for f in classes_files[classname][shard_i*pack_size:(shard_i+1)*pack_size]:
    #         filenames_for_shard[shard_i].append(f)

    # [np.random.shuffle(filenames_for_shard[i]) for i in range(nb_models)]

    # for shard_i in range(nb_model):
    # record_file = f'images_{shard_i}.tfrecords'
    # with tf.io.TFRecordWriter(record_file) as writer:
    #     for f in filenames_for_shard[shard_i]:
    #     img = cv2.imread(f)[...,::-1]
    #     img = tf.image.resize_with_crop_or_pad(img, 224, 224)
    #     img = arr(img)
    #     cv2.imwrite('temp/img.jpg', img)
    #     img_string = open('temp/img.jpg', 'rb').read()

    #     tf_example = create_tf_feature(img_string, class_id)
    #     writer.write(tf_example.SerializeToString())

