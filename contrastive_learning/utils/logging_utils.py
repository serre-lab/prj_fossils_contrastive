"""
logging_utils.py

Created by: Jacob A Rose
Created On: Monday, March 15th, 2021

Contains:

func get_datetime_str(datetime_obj: datetime.datetime=None)
func np_onehot(y: np.ndarray, depth: int)
func test_np_onehot()

class PredictionResults(self, y_prob, y_true, class_names=None, name='predictions', enforce_schema=True, **extra_metadata)
class PredictionMetrics(self, results: PredictionResults, name: str='metrics')

func get_predictions(x: np.ndarray, y: np.ndarray, model: tf.keras.models.Model) -> [np.ndarray, np.ndarray, np.ndarray]
func get_hardest_k_examples(x: np.ndarray,
                            y: np.ndarray,
                            model: tf.keras.models.Model,
                            k: int=32
                            ) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]
func log_high_loss_examples(test_dataset, model, k=32, log_predictions=True, max_rows=10000, run=None, commit=False)

"""

import datetime
import numpy as np
from typing import Dict
from boltons.dictutils import OneToOne
import os
import pandas as pd
from pathlib import Path
import wandb
import tensorflow as tf
from typing import List, Any, Dict, Union
from contrastive_learning.data import stateful
from contrastive_learning.utils.data_utils import log_model_artifact
from contrastive_learning.utils.plotting_utils import display_classification_report



def get_datetime_str(datetime_obj: datetime.datetime=None):
    '''Helper function to get formatted date and time as a str. Defaults to current time if none is passed.
    
    Example:
        >> get_datetime_str()
        'Thu Dec 10 04:10:31 2020'
    '''
    
    if datetime_obj is None:
        datetime_obj = datetime.datetime.utcnow()
    return datetime_obj.strftime('%c')

def np_onehot(y: np.ndarray, depth: int):
    '''Stand in replacement for tf.onehot().numpy()    
    '''
    return np.identity(depth)[y].astype(np.uint8)

def test_np_onehot():
    y = np.array([0,4,2,1,3])
    y_onehot_ground = np.array([[1., 0., 0., 0., 0.],
                                [0., 0., 0., 0., 1.],
                                [0., 0., 1., 0., 0.],
                                [0., 1., 0., 0., 0.],
                                [0., 0., 0., 1., 0.]])
    y_one_hot = np_onehot(y, depth=5)
    
    assert np.all(y_one_hot == y_onehot_ground)
    assert np.all(np.argmax(y_one_hot, axis=1) == y)


class PredictionResults(stateful.Stateful):
    ''' Container for validating, saving, and loading data and metadata associated with predictions made by a tensorflow image classification model
    
    
    # TODO:
        1. Implement __getitem__, __setitem__ methods to allow selecting individual examples with the [] overloaded operator,
            e.g.
            Integer slicing:
                >>prediction_results[2]
                (y_true[2], y_pred[2])
            Query sample IDs by str:
                >>prediction_results['Wing_234']
                (y_true[i], y_pred[i])
                where i == argwhere(sample ID=='Wing_234')
            
        2. Implement  __contains__ method to enable the ability to check if a particular sample is contained by searching for a unique sample ID
            e.g.
            >>'Wing_234' in prediction_results
            False
    
    '''
    
    name: str = 'predictions'
    _y_prob: np.ndarray = None
    _y_true: np.ndarray = None
    class_names: List[str] = None
    _extra_metadata: Dict[str,Any] = None
        
    def __init__(self,
                 y_prob: np.ndarray,
                 y_true: np.ndarray,
                 class_names: Union[List, np.ndarray]=None,
                 name: str='predictions',
                 enforce_schema: bool=True,
                 **extra_metadata):
        """
        PredictionResults data container

        N = # of samples
        M = # of classes

        Args:
            y_prob (np.ndarray):
                Model output probabilities (dtype==float) prior to performing argmax to get predicted labels. Array shape == (N,M)
            y_true (np.ndarray): 
                One-hot encoded ground truth labels (dtype==uint8) for supervised learning problems. Array shape == (N,M)
            class_names (Union[List, np.ndarray], optional): Defaults to None.
                List-like container for the string class names, for each of which their index in the sequence corresponds to their integer encoded values.
            name (str, optional): Defaults to 'predictions'.
            enforce_schema (bool, optional): Defaults to True.
                If False, skip executing the data validation tests in PredictionResults.enforce_schema()
        """                 
        self._assign_values(y_prob=y_prob, y_true=y_true, class_names=class_names, name=name, enforce_schema=enforce_schema, **extra_metadata)
        
    def _assign_values(self, 
                       y_prob,
                       y_true,
                       class_names=None,
                       name='predictions',
                       enforce_schema=True,
                       **extra_metadata):
        self.y_prob = y_prob # (N,M)
        self.y_true = y_true # (N,M) one_hot encoded
        self.class_names = class_names #list of len == M
        self.name = name
        self.extra_metadata = extra_metadata
        if enforce_schema:
            self._enforce_schema()
        
    def _enforce_schema(self):
        assert self.y_prob is not None
        assert self.y_true is not None
        assert isinstance(self.y_prob, np.ndarray)
        assert isinstance(self.y_true, np.ndarray)
        assert self.y_prob.ndim == self.y_true.ndim == 2
        assert self.y_prob.shape == self.y_true.shape
        assert self.y_prob.shape[0] == self.num_samples
        if self.class_names:
            assert len(self.class_names) == self.num_classes
            
    @property
    def extra_metadata(self):
        return self._extra_metadata

    @extra_metadata.setter
    def extra_metadata(self, metadata: Dict):
        if len(metadata) > 0:
            assert isinstance(metadata, dict)
            for key, item in metadata.items():
                if isinstance(item, np.integer):
                    self._extra_metadata.update({key:int(item)})
                elif isinstance(item, np.floating):
                    self._extra_metadata.update({key:float(item)})
                elif isinstance(item, np.ndarray):
                    self._extra_metadata.update({key:item.tolist()})

    def decode_names(self, y: np.ndarray, as_array: bool=False):
        ''' int -> str data labels/predictions
        Input an array of integer labels to get their corresponding str names as either a list (default) or a np.ndarray.
        
        '''
#         y_true = self.get_y_true(one_hot=False)
        names = [self.class_names[i] for i in y]
        if as_array:
            return np.asarray(names)
        return names
    ####################################
    def get_y_pred(self, one_hot=False):
        if one_hot:
            return np_onehot(self.y_pred, depth=self.num_classes)
        return self.y_pred
                    
    def get_y_true(self, one_hot=True):
        if one_hot:
            return self.y_true
        return np.argmax(self.y_true, axis=1)
        
    @property
    def y_pred(self):
        return np.argmax(self.y_prob, axis=1)
    
    @property
    def y_true(self):
        return self._y_true
    
    @y_true.setter
    def y_true(self, y_true_new):
        ''' Constrains any new data assigned to self.y_true is formatted as ONE-HOT-ENCODED with shape (N,M)
        Keep default format of instance's y_true as one_hot encoding while in memory, convert to sparse int encoding for serialization to disk.
        
        '''
        if y_true_new.ndim == 1:
            if not issubclass(type(y_true_new[0]), np.integer):
                y_true_new = y_true_new.astype(np.uint8)
            y_true_new = tf.one_hot(y_true_new, depth=self.num_classes).numpy()
        self._y_true = y_true_new

    ####################################
    @property
    def num_classes(self):
        return self.y_prob.shape[1]

    @property
    def num_samples(self):
        return self.y_prob.shape[0]
    
    
    def get_state(self):
        y_true = self.get_y_true(one_hot=False)
        y_prob = self.y_prob
        state = {'meta':{
                         'name':self.name,
                         'num_classes':self.num_classes,
                         'num_samples':self.num_samples,
                         **{k:v for k,v in self.extra_metadata.items()}
                 },
                 'data':{
                         'class_names':self.class_names,
                         'y_true':y_true.tolist(), #Store more memory efficient representation of y_true
                         'y_prob':y_prob.tolist()
                 }}
        assert np.allclose(y_true, np.asarray(state['data']['y_true']))
        assert np.allclose(y_prob, np.asarray(state['data']['y_prob']))
        
        return state
    
    def set_state(self, state):        
        for key in ['class_names','y_true','y_prob']:
            assert key in state['data'].keys()
            
        y_prob = np.asarray(state['data']['y_prob'])
        y_true = np.asarray(state['data']['y_true'])
        class_names =  state['data']['class_names']
        name = state['meta']['name']
        extra_metadata = {k:v for k,v in state['meta'].items() if k not in ['name','num_classes','num_samples']}
        self._assign_values(y_prob=y_prob, y_true=y_true, class_names=class_names, name=name, **extra_metadata)
        
        assert self.num_classes == state['meta']['num_classes']
        assert self.num_samples == state['meta']['num_samples']
        assert self.name == state['meta']['name']
        
    def __repr__(self):
        return '\n'.join([f'{k}:\n\t{v}' for k,v in self.get_state()['meta'].items()])
        
    @classmethod
    def log_model_artifact(cls,
                           model: tf.keras.models.Model,
                           model_path: str, 
                           class_encoder: Union[List, np.ndarray]=None,
                           run=None,
                           metadata: Dict[Any]=None,
                           name: str=None):
        """
        Logs a trained model as an artifact to wandb

        Args:
            model (tf.keras.models.Model): [description]
            model_path (str): [description]
            class_encoder (Union[List, np.ndarray], optional): [description]. Defaults to None.
            run ([type], optional): [description]. Defaults to None.
            metadata (Dict[Any], optional): [description]. Defaults to None.
            name (str, optional): [description]. Defaults to None.
        """                           
        metadata = metadata or {}
        name = name or ''
        print(f'Logging model artifact for Object {name} at\n{model_path}')
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        run = run or wandb
        log_model_artifact(model, model_path, encoder=class_encoder, run=run, metadata=metadata)

        

        
        
        
        

class PredictionMetrics(stateful.Stateful):
    """
    PredictionMetrics class

    Container for 

    Args:
        results (PredictionResults): [description]
    """    
    name: str = 'metrics'
    _results: PredictionResults = None
    class_names: List[str] = None
    
    _metric_names = ['tp','tn','fp','fn']
    _metric_agg_funcs = ['sum','sum','sum','sum']
    _agg_func = {'sum':np.sum, 'mean':np.mean, 'std':np.std}
    

    def __init__(self, results: PredictionResults, name='metrics'):
        self.results = results
        self.class_names = results.class_names
        
    def get_values(self, agg_mode: str=None):
        '''
        if agg_mode in ['sample',0]: Return per-sample metrics
            out: np.ndarray with shape == (num_samples,)
        elif agg_mode in ['class',1]: Return local per-class metrics
            out: np.ndarray with shape == (num_classes,)
        elif agg_mode in ['macro',2]: Return global scalar metrics produced by first calculating per-class values with agg_mode 1, then performing the same aggregation over all classes
            out: np.ndarray with shape == (1,)
        elif agg_mode in ['micro',3]: Return global scalar metrics produced by first calculating per-sample values with agg_mode 0, then performing the same aggregation over all samples equally
            out: np.ndarray with shape == (1,)
        else: Return raw onehot metrics without aggregation
            out: np.ndarray with shape (num_samples, num_classes )
        
        '''
        values = {'tp':self.tp,
                  'tn':self.tn,
                  'fp':self.fp,
                  'fn':self.fn}
#         mean_values = {'recall':self.recall(),
#                        'precision':self.precision(),
#                        'accuracy':self.accuracy()}
        
        if agg_mode in ['sample',0]:
            return {k:np.sum(v, axis=1) for k,v in values.items()}
        
        elif agg_mode in ['class',1]:
            return {k:np.sum(v, axis=0) for k,v in values.items()}
            
        elif agg_mode in ['macro',2]:
            return {k:np.sum(v) for k,v in self.get_values(agg_mode='class').items()}

        elif agg_mode in ['micro',3]:
            return {k:np.sum(v) for k,v in values.items()}
        
        else:
            return values
        
    def classification_report(self, agg_funcs=['micro', 'macro'], display_widget=False):
        if agg_funcs == 'class':
            agg = agg_funcs
            classification_report = pd.DataFrame({'recall':self.recall(agg),
                                          'precision':self.precision(agg),
                                          'accuracy':self.accuracy(agg),
                                          'f1-Score':self.f1(agg)},
                                          index=self.class_names)
        else:
            classification_report = []
            for agg in agg_funcs:
                classification_report.append({'recall':self.recall(agg),
                                              'precision':self.precision(agg),
                                              'accuracy':self.accuracy(agg),
                                              'f1-Score':self.f1(agg)})
            classification_report = pd.DataFrame.from_records(classification_report)
            classification_report.index = agg_funcs
            
        if display_widget:
            try:
                classification_report = display_classification_report(classification_report)
            except Exception as e:
                print(e)
                import ipdb;
                ipdb.set_trace()
                print('Failed to display HTML widget. Returning classification report dataframe')
                
        return classification_report
        
    @property
    def results(self):
        return self._results

    @results.setter
    def results(self, results: PredictionResults):
        assert isinstance(results, PredictionResults)
        self._results = results
        self.y_pred_onehot = results.get_y_pred(one_hot=True)
        self.y_true_onehot = results.get_y_true(one_hot=True)

    @property
    def tp(self): # -> shape==(N,M)
        return np.logical_and(self.y_pred_onehot == 1, self.y_true_onehot == 1)
    @property
    def tn(self):# -> shape==(N,M)
        return np.logical_and(self.y_pred_onehot == 0, self.y_true_onehot == 0)
    @property
    def fp(self):# -> shape==(N,M)
        return np.logical_and(self.y_pred_onehot == 1, self.y_true_onehot == 0)
    @property
    def fn(self):# -> shape==(N,M)
        return np.logical_and(self.y_pred_onehot == 0, self.y_true_onehot == 1)
    
    def recall(self, agg_mode: str=None):
        
        if agg_mode in ['micro',3]:
            values = self.get_values('micro')
        else:
            values = self.get_values('class')
            
        tp, tn, fp, fn =  values['tp'], values['tn'], values['fp'], values['fn']
        recall = tp / (tp + fn)
        
        if agg_mode in ['macro',2]:
            recall = np.mean(recall)
        return recall #{k:np.mean(v) for k,v in values.items()}

    def precision(self, agg_mode: str=None):
        
        if agg_mode in ['micro',3]:
            values = self.get_values('micro')
        else:
            values = self.get_values('class')
            
        tp, tn, fp, fn =  values['tp'], values['tn'], values['fp'], values['fn']
        precision = tp / (tp + fp)
        if agg_mode in ['macro',2]:
            precision = np.mean(precision)
        return precision #{k:np.mean(v) for k,v in values.items()}

    def accuracy(self, agg_mode: str=None):
        
        if agg_mode in ['micro',3]:
            values = self.get_values('micro')
        else:
            values = self.get_values('class')
            
        tp, tn, fp, fn =  values['tp'], values['tn'], values['fp'], values['fn']
        accuracy = (tp + tn) / (tp + tn + tp + fp)
        if agg_mode in ['macro',2]:
            accuracy = np.mean(accuracy)
        return accuracy #{k:np.mean(v) for k,v in values.items()}

    
    def f1(self, agg_mode: str=None):
        
        recall = self.recall(agg_mode)
        precision = self.precision(agg_mode)
        
        f1 = 2 * (precision * recall) / (precision + recall)
        
        return f1

    @property
    def num_classes(self):
        return self.results.num_classes
        
    @property
    def positives(self):
        positives = self.tp + self.fn
        assert positives.shape[0] == positives.sum()
        return positives

    @property
    def negatives(self):
        negatives = self.tn + self.fp
        assert (negatives.shape[0]*(self.num_classes-1)) == negatives.sum()
        return negatives


##################################################################
##################################################################


def get_predictions(x: np.ndarray,
                    y: np.ndarray,
                    model: tf.keras.models.Model
                    ) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Input the input images x, true labels y, and compiled model with self.predict() method
    Return the logits, predicted class labels as ints, and the per sample losses

    Args:
        x (np.ndarray): [description]
        y (np.ndarray): [description]
        model (tf.keras.models.Model): [description]

    Returns:
        [np.ndarray, np.ndarray, np.ndarray]: [description]
    """    
    y_prob = model.predict(x)
    y_pred = np.argmax(y_prob, axis=1)
    losses = tf.keras.losses.categorical_crossentropy(y, y_prob)
    
    return y_prob, y_pred, losses


def get_hardest_k_examples(x: np.ndarray, 
                           y: np.ndarray,
                           model: tf.keras.models.Model,
                           k: int=32) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Input the input images x, true labels y, and compiled model with self.predict() method
    Return the hardest/highest k losses, images, true labels, and predicted labels

    Args:
        x (np.ndarray): [description]
        y (np.ndarray): [description]
        model (tf.keras.models.Model): [description]
        k (int, optional): [description]. Defaults to 32.

    Returns:
        highest_k_losses: np.ndarray
        hardest_k_examples: np.ndarray
        hardest_k_true_labels: np.ndarray
        hardest_k_predictions: np.ndarray
    """

    _, y_pred, losses = get_predictions(x, y, model)

    argsort_loss =  np.argsort(losses)[::-1]
    highest_k_losses = np.array(losses)[argsort_loss[:k]]
    hardest_k_examples = np.stack([x[i,...] for i in argsort_loss[:k]])
    hardest_k_true_labels = np.argmax(np.stack([y[i] for i in argsort_loss[:k]]), axis=1)
    hardest_k_predictions = np.stack([y_pred[i] for i in argsort_loss[:k]])

    return highest_k_losses, hardest_k_examples, hardest_k_true_labels, hardest_k_predictions


def _get_1_epoch_from_tf_data(dataset: tf.data.Dataset, max_rows: int=np.inf) -> [np.ndarray, np.ndarray]:
    """
    Get up to the maximum of either len(dataset) or max_rows batches yielded from dataset and stacked into np.ndarrays

    Args:
        dataset (tf.data.Dataset): [description]
        max_rows (int, optional): [description]. Defaults to np.inf.

    Returns:
        [np.ndarray, np.ndarray]: [description]
    """    
    bsz = next(iter(dataset))[0].shape[0]
    steps = len(dataset)*bsz
    steps = min([steps, max_rows])
    dataset = dataset.unbatch()
    print(steps)
    x, y = next(iter(dataset.batch(steps).take(1)))
    return x, y

        
def log_high_loss_examples(test_dataset: tf.data.Dataset,
                           model: tf.keras.models.Model,
                           k: int=32,
                           log_predictions: bool=True,
                           max_rows: int=10000,
                           run=None,
                           commit: bool=False) -> None:
    """

    Args:
        test_dataset (tf.data.Dataset): [description]
        model (tf.keras.models.Model): [description]
        k (int, optional): [description]. Defaults to 32.
        log_predictions (bool, optional): [description]. Defaults to True.
        max_rows (int, optional): [description]. Defaults to 10000.
        run ([type], optional): [description]. Defaults to None.
        commit (bool, optional): [description]. Defaults to False.
    """    
    print(f'logging k={k} hardest examples')
    x, y_true = _get_1_epoch_from_tf_data(test_dataset, max_rows=max_rows)
    highest_k_losses, hardest_k_examples, hardest_k_true_labels, hardest_k_predictions = get_hardest_k_examples(x, 
                                                                                                                y_true,
                                                                                                                model,
                                                                                                                k=k)
    run = run or wandb
    if log_predictions:
        _, y_pred, _ = get_predictions(x,
                                       y_true,
                                       model)
        max_rows = min([int(max_rows), len(y_pred)])
        print(f'logging {max_rows} true & predicted integer labels')
        y_true, y_pred = y_true[:max_rows,...], y_pred[:max_rows] #wandb rate limits allow a max of 10,000 rows
        data_table = pd.DataFrame({'y':np.argmax(y_true,axis=1),'y_pred':y_pred})
        table = wandb.Table(dataframe=data_table)
        run.log({"test_data" : table}, commit=False) #wandb.plot.bar(table, "label", "value", title="Custom Bar Chart")
        
    run.log(
        {"high-loss-examples":
                            [wandb.Image(hard_example, caption = f'true:{label},\npred:{pred}\nloss={loss}')
                             for hard_example, label, pred, loss in zip(hardest_k_examples, hardest_k_true_labels, hardest_k_predictions, highest_k_losses)]
        }, commit=commit)

##################################################################
# END
##################################################################





































##################################################################
##################################################################

# def get_hardest_k_examples(test_dataset, model, k=32):
#     class_probs = model.predict(test_dataset)
#     predictions = np.argmax(class_probs, axis=1)
#     losses = tf.keras.losses.categorical_crossentropy(test_dataset.y, class_probs)
#     argsort_loss =  np.argsort(losses)

#     highest_k_losses = np.array(losses)[argsort_loss[-k:]]
#     hardest_k_examples = test_dataset.x[argsort_loss[-k:]]
#     true_labels = np.argmax(test_dataset.y[argsort_loss[-k:]], axis=1)

#     return highest_k_losses, hardest_k_examples, true_labels, predictions
        
# def log_high_loss_examples(test_dataset, model, k=32, run=None):
#     print(f'logging k={k} hardest examples')
#     losses, hardest_k_examples, true_labels, predictions = get_hardest_k_examples(test_dataset, model, k=k)
    
#     run = run or wandb
#     run.log(
#         {"high-loss-examples":
#                             [wandb.Image(hard_example, caption = f'true:{label},\npred:{pred}\nloss={loss}')
#                              for hard_example, label, pred, loss in zip(hardest_k_examples, true_labels, predictions, losses)]
#         })


    

##############################################################################
# prediction_results = PredictionResults(y_prob,
#                                        y_true,
#                                        class_names=class_names,
#                                        name='test_predictions',
#                                        dataset_name=dataset_name,
#                                        creation_date=get_datetime_str(), 
#                                        model_name = resnet.model_name,
#                                        groups=resnet.groups)

# prediction_results.save(fname=os.path.join(save_dir,'prediction_results.json'))
# prediction_results.reload(fname=os.path.join(save_dir,'prediction_results.json'))

# import wandb
# run = wandb.init()
# prediction_results.log_json_artifact(path=os.path.join(save_dir,'prediction_results.json'), run=run) #, artifact_type: str=None)

# artifact = run.use_artifact('jrose/genetic_algorithm-Notebooks/prediction_results.json:v0', type="<class '__main__.PredictionResults'>")
# artifact_dir = artifact.download()
    
#     def agg_by_sample(self, metric)
#         assert np.sum([tp,tn,fp,fn]) == np.prod(self.y_pred_onehot.shape)

#         # per-sample metrics
#         self.tp = np.sum(np.logical_and(y_pred_onehot == 1, y_true_onehot == 1), axis=1)
#         # True Negative (TN): we predict a label of 0 (negative), and the true label is 0.
#         self.tn = np.sum(np.logical_and(y_pred_onehot == 0, y_true_onehot == 0), axis=1)
#         # False Positive (FP): we predict a label of 1 (positive), but the true label is 0.
#         self.fp = np.sum(np.logical_and(y_pred_onehot == 1, y_true_onehot == 0), axis=1)
#         # False Negative (FN): we predict a label of 0 (negative), but the true label is 1.
#         self.fn = np.sum(np.logical_and(y_pred_onehot == 0, y_true_onehot == 1), axis=1)

#         assert np.sum([tp,tn,fp,fn]) == np.prod(y_pred_onehot.shape)