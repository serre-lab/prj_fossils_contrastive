import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from contrastive_learning.data.data_utils import load_dataset_from_artifact, class_counts

# def _normalize(x, y):
#     return x.astype('float32') / 255.0, tf.one_hot(y[:,0], 10).numpy()


class ClassLabelEncoder(stateful.Stateful):
    def __init__(self, true_labels: np.ndarray, num_classes: int=None, name: str=''):
        self.dataset_name = name
        self.class_names = class_counts(true_labels)
        
        self.num_samples = true_labels.shape[0]
        self.num_classes = num_classes or len(self.class_names)
        self._str2int = {name:num for num, name in enumerate(self.class_names)}
        self._int2str = {num:name for num, name in enumerate(self.class_names)}
        
        
    def __getstate__(self):
        return {'dataset_name':self.dataset_name,
                'num_samples':self.num_samples,
                'num_classes':self.num_classes,
                'class_names':self.class_names}.copy()

    def __setstate__(self, state):
        self.__dict__.update({'dataset_name':state['dataset_name'],
                              'num_samples':state['num_samples'],
                              'num_classes':state['num_classes'],
                              'class_names':state['class_names']})

        self._str2int = {name:num for num, name in enumerate(state['class_names'])}
        self._int2str = {num:name for num, name in enumerate(state['class_names'])}
        self.info = None
        
    def get_state(self):
        return self.__getstate__()
    
    def set_state(self, state):
        self.__setstate__(state)

    def decode_predictions(self, preds, top=5):
        """Decodes the prediction of an PlantVillage model.
        Arguments:
            preds: Numpy array encoding a batch of predictions.
            top: Integer, how many top-guesses to return. Defaults to 5.
        Returns:
            A list of lists of top class prediction tuples
            `(class_name, class_description, score)`.
            One list of tuples per sample in batch input.
        Raises:
            ValueError: In case of invalid shape of the `pred` array
            (must be 2D).
        """
        if preds.ndim != 2 or preds.shape[1] != self.num_classes:
            raise ValueError(f'`decode_predictions` expects '
                             'a batch of predictions '
                            f'(i.e. a 2D array of shape (samples, {self.num_classes})). '
                             'Found array with shape: ' + str(preds.shape))
        results = []
        for pred in preds:
            top_indices = pred.argsort()[-top:][::-1]
            result = [tuple(self.class_names[str(i)]) + (pred[i],) for i in top_indices]
            result.sort(key=lambda x: x[2], reverse=True)
            results.append(result)
        return results


    def str2int(self, labels: Union[List[str],Tuple[str]]):
        labels = self._valid_eager_tensor(labels)
        if not isinstance(labels, (list, tuple)):
            if isinstance(labels, pd.Series):
                labels = labels.values
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()
            else:
                assert isinstance(labels, str)
                labels = [labels]
        output = []
        keep_labels = self._str2int
        for l in labels:
            if l in keep_labels:
                output.append(keep_labels[l])
        return output
#         return [self._str2int(l) for l in labels]

    def int2str(self, labels: Union[List[int],Tuple[int]]):
        labels = self._valid_eager_tensor(labels)
        if not isinstance(labels, (list, tuple)):
            if isinstance(labels, np.ndarray):
                labels = labels.tolist()
            else:
                assert isinstance(labels, (int, np.int64))
                labels = [labels]
        output = []
        keep_labels = self._int2str
        for l in labels:
            if l in keep_labels:
                output.append(keep_labels[l])
        return output
#         return [self._int2str(l) for l in labels]

    def one_hot(self, label: tf.int64):
        '''
        One-Hot encode integer labels
        Use tf.data.Dataset.map(lambda x,y: (x, encoder.one_hot(y))) and pass in individual labels already encoded in int64 format.
        '''
        return tf.one_hot(label, depth=self.num_classes)

    def __repr__(self):
        return f'''Dataset Name: {self.dataset_name}
        Num_samples: {self.num_samples}
        Num_classes: {self.num_classes}'''

    def _valid_eager_tensor(self, tensor, strict=False):
        '''
        If tensor IS an EagerTensor, return tensor.numpy(). 
        if strict==True, and tensor IS NOT an EagerTensor, then raise AssertionError.
        if strict==False, and tensor IS NOT an EagerTensor, then return tensor without modification 
        '''
        try:
            assert isinstance(tensor, tf.python.framework.ops.EagerTensor)
            tensor = tensor.numpy()
        except AssertionError:
            if strict:
                raise AssertionError(f'Strict EagerTensor requirement failed assertion test in ClassLabelEncoder._valid_eager_tensor method')
#         np_tensor = tensor.numpy()
        return tensor




def load_data_from_tensor_slices(data: pd.DataFrame,
                                 cache_paths: Union[bool,str]=True,
                                 training=False,
                                 seed=None,
                                 x_col='path',
                                 y_col='label',
                                 dtype=None):
    dtype = dtype or tf.uint8
    num_samples = data.shape[0]

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
                                     'y':example['y']}, num_parallel_calls=-1)
    return data


def load_pnas_dataset(threshold=100,
                      validation_split=0.1,
                      seed=None,
                      y='family'):

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
    class_encoder = ClassLabelEncoder(true_labels=data['train'][y], name='PNAS')
    
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


def get_unsupervised(batch_size=128, val_split=0.2):
    data, _ = load_and_extract_pnas(threshold=100,
                                    validation_split=0.2,
                                    seed=None,
                                    x_col='path',
                                    y_col='family')

    train_dataset = data['train'].map(lambda x,y: x).batch(batch_size)
    val_dataset = data['val'].map(lambda x,y: x).batch(batch_size)
    test_dataset = data['test'].map(lambda x,y: x).batch(batch_size)

    return train_dataset, val_dataset, test_dataset


def get_supervised(batch_size=128, val_split=0.2):
    data, _ = load_and_extract_pnas(threshold=100,
                                    validation_split=0.2,
                                    seed=None,
                                    x_col='path',
                                    y_col='family')

    train_dataset = data['train'].batch(batch_size)
    val_dataset = data['val'].batch(batch_size)
    test_dataset = data['test'].batch(batch_size)

    return train_dataset, val_dataset, test_dataset