import numpy as np
import pandas as pd
import tensorflow as tf
from typing import Union, Tuple, List, Dict
from sklearn.model_selection import train_test_split

from contrastive_learning.utils.data_utils import load_dataset_from_artifact, class_counts
from contrastive_learning.utils.label_utils import ClassLabelEncoder
from contrastive_learning.data import stateful
# def _normalize(x, y):
#     return x.astype('float32') / 255.0, tf.one_hot(y[:,0], 10).numpy()


PNAS_root_dir = "/media/data_cifs/projects/prj_fossils/data/processed_data/data_splits/PNAS_family_100"

def load_data_from_tensor_slices(data: pd.DataFrame,
                                 cache_paths: Union[bool,str]=True,
                                 training: bool=False,
                                 seed: int=None,
                                 x_col: str='path',
                                 y_col: str='label',
                                 dtype=None):
    dtype = dtype or tf.uint8
    num_samples = data.shape[0]
    num_classes = len(set(data[y_col].values))

    def load_img(image_path):
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)
        return img

    x_data = tf.data.Dataset.from_tensor_slices(data[x_col].values.tolist())
    y_data = tf.data.Dataset.from_tensor_slices(data[y_col].values.tolist())
    
    data = tf.data.Dataset.zip((x_data, y_data))
    data = data.map(lambda x, y: {'x':x,'y':y})
    data = data.take(num_samples).cache()
    
    # TODO TEST performance and randomness of the order of shuffle and cache when shuffling full dataset each iteration, but only filepaths and not full images.
    if training:
        data = data.shuffle(num_samples,seed=seed, reshuffle_each_iteration=True)

    data = data.map(lambda example: {'x':tf.image.convert_image_dtype(load_img(example['x'])*255.0,dtype=dtype),
                                     'y':tf.one_hot(example['y'], num_classes)}, num_parallel_calls=-1)
    return data
# def _normalize(x, y):
#     return x.astype('float32'), tf.one_hot(y[:,0], 10).numpy()

def load_pnas_dataset(threshold=100,
                      validation_split=0.2,
                      seed=None,
                      y='family'
                      ) -> Dict[str,pd.DataFrame]:
    """[summary]

    Args:
        threshold (int, optional): [description]. Defaults to 100.
        validation_split (float, optional): [description]. Defaults to 0.2.
        seed ([type], optional): [description]. Defaults to None.
        y (str, optional): [description]. Defaults to 'family'.

    Returns:
        Dict[str,pd.DataFrame]: [description]
    """

    train_df, test_df = load_dataset_from_artifact(dataset_name='PNAS', threshold=threshold, test_size=0.5, version='latest')
    train_df, val_df  = train_test_split(train_df, test_size=validation_split, random_state=seed, shuffle=True, stratify=train_df[y])
    
    return {'train':train_df,
            'val':val_df,
            'test':test_df}

def extract_data(data: Dict[str,pd.DataFrame],
                 x='path',
                 y='family',
                 shuffle_first=True,
                 data_cifs_repair=True,
                 seed=None):
    
    subset_keys = list(data.keys())
    class_encoder = ClassLabelEncoder(y_true=data['train'][y], name='PNAS')
    
    extracted_data = {}
    for subset in subset_keys:
        if shuffle_first:
            data[subset] = data[subset].sample(frac=1)
            
        if data_cifs_repair:
            data[subset] = data[subset].assign(raw_path=data[subset].apply(lambda x: x.raw_path.replace('data_cifs_lrs','data_cifs'), axis=1),
                                               path=data[subset].apply(lambda x: x.path.replace('data_cifs_lrs','data_cifs'), axis=1))
        
        paths = data[subset][x]
        text_labels = data[subset][y]
        labels = class_encoder.str2int(text_labels)
        
        extracted_data[subset] = pd.DataFrame.from_records([{'path':path, 'label':label, 'text_label':text_label} for path, label, text_label in zip(paths, labels, text_labels)])
        
        training = (subset=='train')
        extracted_data[subset] = load_data_from_tensor_slices(data=extracted_data[subset], training=training, seed=seed, x_col='path', y_col='label', dtype=tf.float32)
    
    return extracted_data, class_encoder



def load_and_extract_pnas(threshold=100,
                          validation_split=0.2,
                          seed=None,
                          x_col='path',
                          y_col='family'):
    

    data = load_pnas_dataset(threshold=threshold,
                             validation_split=validation_split,
                             seed=seed,
                             y=y_col)

    data, class_encoder = extract_data(data=data,
                                     x=x_col,
                                     y=y_col,
                                     shuffle_first=True,
                                     seed=seed)
    
    return data, class_encoder


def get_unsupervised(batch_size: int=1,
                     val_split=0.2,
                     threshold=100,
                     seed: int=None):

    data, _ = load_and_extract_pnas(threshold=threshold,
                                    validation_split=val_split,
                                    seed=seed)

    train_dataset = data['train'].map(lambda sample: sample['x'])#.batch(batch_size)
    val_dataset = data['val'].map(lambda sample: sample['x'])#.batch(batch_size)
    test_dataset = data['test'].map(lambda sample: sample['x'])#.batch(batch_size)

    return train_dataset, val_dataset, test_dataset


def get_supervised(batch_size: int=1,
                   val_split=0.2,
                   threshold=100,
                   seed: int=None,
                   return_label_encoder: bool=False):

    data, label_encoder = load_and_extract_pnas(threshold=threshold,
                                    validation_split=val_split,
                                    seed=seed)

    train_dataset = data['train']#.batch(batch_size)
    val_dataset = data['val']#.batch(batch_size)
    test_dataset = data['test']#.batch(batch_size)


    if return_label_encoder:

        return (train_dataset, val_dataset, test_dataset), label_encoder

    return train_dataset, val_dataset, test_dataset









# def get_unsupervised(val_split=0.2, seed: int=None):
#     data, _ = load_and_extract_pnas(threshold=100,
#                                     validation_split=val_split,
#                                     seed=seed,
#                                     x_col='path',
#                                     y_col='family')

#     train_dataset = data['train'].map(lambda sample: sample['x']) #.batch(batch_size)
#     val_dataset = data['val'].map(lambda sample: sample['x']) #.batch(batch_size)
#     test_dataset = data['test'].map(lambda sample: sample['x']) #.batch(batch_size)

#     return train_dataset, val_dataset, test_dataset


# def get_supervised(val_split=0.2, seed: int=None):
#     data, _ = load_and_extract_pnas(threshold=100,
#                                     validation_split=val_split,
#                                     seed=seed,
#                                     x_col='path',
#                                     y_col='family')

#     train_dataset = data['train'] #.batch(batch_size)
#     val_dataset = data['val'] #.batch(batch_size)
#     test_dataset = data['test'] #.batch(batch_size)

#     return train_dataset, val_dataset, test_dataset