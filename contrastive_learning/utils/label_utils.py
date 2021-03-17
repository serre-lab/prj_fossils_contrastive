"""
label_utils.py

Created by: Jacob A Rose
Created On: Monday, March 15th, 2021

Contains:

class ClassLabelEncoder
func save_class_labels(class_labels: Union[ClassLabelEncoder,dict,OneToOne], label_path: str)
func load_class_labels(label_path: str)


"""

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_datasets as tfds
from typing import Union, List, Tuple
from boltons.dictutils import OneToOne
from contrastive_learning.data import stateful, data_utils
from contrastive_learning.utils import data_utils


class ClassLabelEncoder(stateful.Stateful):
    def __init__(self,
                 y_true: np.ndarray,
                 num_classes: int=None,
                 name: str='',
                 alphabetical=True):
        """ Creates custom object for manipulating class labels throughout pipeline

        Note: Object does not permanently store original y_true for efficiency. 
        They're provided at instantiation and used to calculate summary statistics & create string->int mappings.

        Args:
            y_true (np.ndarray): 
                Array containing class labels as strings.
            num_classes (int, optional): Defaults to None.
                TODO: Currently, manually providing this only stores the number but doesn't impact whether any classes are kept.
            name (str, optional): Defaults to ''.
                Useful to provide a name attribute if working with multiple encoders
        """        
        self.dataset_name = name
        self.class_names = data_utils.class_counts(y_true)
        if alphabetical:
            self.class_names = sorted(self.class_names)
        
        self.num_samples = y_true.shape[0]
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




def save_class_labels(class_labels: Union[ClassLabelEncoder,dict,OneToOne], label_path: str):
    '''
    Save dictionary of str:int class labels as a csv file containing just the str labels in the order they're provided. Use load_class_labels() with the same filepath to load them back.
    
    '''
    if isinstance(class_labels, ClassLabelEncoder):
        label_path+='.json'
        class_labels.save(label_path)
    
    elif isinstance(class_labels, (dict,OneToOne)):
        data = pd.DataFrame(list(class_labels.keys()))
        label_path+'.csv'
        data.to_csv(label_path, index=None, header=False)
    else:
        raise Exception(f'unsupported label object of type {type(class_labels)}')
    
    return label_path


def load_class_labels(label_path: str):
    data = pd.read_csv(label_path, header=None, squeeze=True).values.tolist()
    loaded = OneToOne({label:i for i, label in enumerate(data)})
    return loaded