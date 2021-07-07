"""
data_utils.py

Created by: Jacob A Rose
Created On: Monday, March 15th, 2021

Contains:

func class_counts(y: np.ndarray, as_dataframe: bool=False) -> Union[Dict[Union[str,int],int],pd.DataFrame]
func log_model_artifact(model, model_path, encoder, run=None, metadata=None):

"""

import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple, Optional
from boltons.dictutils import OneToOne
import os
import pandas as pd
from pathlib import Path
# import tensorflow as tf
import wandb

from contrastive_learning.utils.label_utils import ClassLabelEncoder, load_class_labels, save_class_labels









###########################################

def read_train_test_csv(root_dir: str) -> Tuple[pd.DataFrame]:
    """
    Read 2 DataFrames into memory by providing a root_dir containing files named "train.csv" and "test.csv"

    Args:
        root_dir (str): [description]

    Returns:
        train_df, test__df (Tuple[pd.DataFrame]): [description]
    """    
    train_df = pd.read_csv(os.path.join(root_dir,'train.csv'),index_col='id')
    test_df = pd.read_csv(os.path.join(root_dir,'test.csv'),index_col='id')    
    return train_df, test_df


def load_train_test_artifact(artifact_uri: str='jrose/uncategorized/Leaves-PNAS:v1',
                             run=None) -> Tuple[pd.DataFrame]:
    """
    Provide an exact URI corresponding to a CSV dataset registered as an Artifact on WandB.

    Args:
        artifact_uri: (str, optional) Defaults to 'jrose/uncategorized/Leaves-PNAS:v1'.
        run: ([type], optional) Defaults to None.

    Returns:
        train_df, test_df: (Tuple[pd.DataFrame])
            Tuple of DataFrames containing at least 2 columns, one for labels and another for image paths.

    """                             
    if run is None:
        # run = wandb.init(reinit=True)
        run = wandb.init(reinit=True, settings=wandb.Settings(start_method="fork"))
    
    artifact = run.use_artifact(artifact_uri, type='dataset')
    artifact_dir = artifact.download()
    print('artifact_dir =',artifact_dir)

    return read_train_test_csv(root_dir=artifact_dir)




def load_dataset_from_artifact(dataset_name: str='Fossil', 
                               threshold: int=4,
                               test_size: float=0.3,
                               version: str='latest',
                               artifact_name: Optional[str]=None,
                               run=None) -> Tuple[pd.DataFrame]:
    """
    Query Serre-Lab Fossil/Leaves dataset collection of CSV databases registered as Artifacts on WandB.com.

    (As of Wednesday March 17th, 2021) These are located in the private paleoai-project located at https://wandb.ai/brown-serre-lab/paleoai-project/artifacts/dataset/

    Args:
        dataset_name (str, optional): [description]. Defaults to 'Fossil'.
        threshold (int, optional): [description]. Defaults to 4.
        test_size (float, optional): [description]. Defaults to 0.3.
        version (str, optional): [description]. Defaults to 'latest'.
        artifact_name (Optional[str], optional): [description]. Defaults to None.
        run ([type], optional): [description]. Defaults to None.

    Returns:
        Tuple[pd.DataFrame]: [description]
    """                               
    train_size = 1 - test_size
    if artifact_name:
        pass
    elif dataset_name=='Fossil':
        artifact_name = f'{dataset_name}_{threshold}_{int(train_size*100)}-{int(100*test_size)}:{version}'
    elif dataset_name=='PNAS':
        artifact_name = f'{dataset_name}_family_{threshold}_50-50:{version}'
    elif dataset_name=='Leaves':
        artifact_name = f'{dataset_name}_family_{threshold}_{int(train_size*100)}-{int(100*test_size)}:{version}'
    elif dataset_name=='Leaves-PNAS':
        artifact_name = f'{dataset_name}_{int(train_size*100)}-{int(100*test_size)}:{version}'
    elif dataset_name in ['Leaves_in_PNAS', 'PNAS_in_Leaves']:
        artifact_name = f'{dataset_name}_{int(train_size*100)}-{int(100*test_size)}:{version}'
            
    artifact_uri = f'brown-serre-lab/paleoai-project/{artifact_name}'
    return load_train_test_artifact(artifact_uri=artifact_uri, run=run)



###########################################



def class_counts(y: np.ndarray, as_dataframe: bool=False) -> Union[Dict[Union[str,int],int],pd.DataFrame]:
    counts = dict(zip(*np.unique(y, return_counts=True)))
    if as_dataframe:
        counts = pd.DataFrame([(k,v) for k,v in counts.items()]).rename(columns={0:'label', 1:'label_count'})
    return counts


def log_model_artifact(model,
                       model_path,
                       encoder,
                       run=None,
                       metadata=None):
    """

    Args:
        model (tf.keras.models.Model):
        model_path (str): 
        encoder (ClassLabelEncoder): 
        run (optional): Defaults to None.
        metadata (Dict, optional): Defaults to None.
    """                       
    # TODO: link the logged model artifact to a logged classification report
    
    model.save(model_path)
    model_name = str(Path(model_path).name).replace('+','.')
    
    run = run or wandb
    artifact = wandb.Artifact(type='model', name=model_name)
    if os.path.isfile(model_path):
        artifact.add_file(model_path, name=model_name)
        model_dir = os.path.dirname(model_path)
        class_label_path = os.path.join(model_dir, 'labels')
    elif os.path.isdir(model_path):
        artifact.add_dir(model_path, name=model_name)
        class_label_path = os.path.join(model_path, 'labels')

    class_label_path = save_class_labels(class_labels=encoder, label_path=class_label_path)
    artifact.add_file(class_label_path, name=str(Path(class_label_path).name))
    run.log_artifact(artifact)











# def load_Leaves_Minus_PNAS_test_dataset():
#     run = wandb.init(reinit=True)
#     with run:
#         artifact = run.use_artifact('jrose/uncategorized/Leaves-PNAS_test:v2', type='dataset')
#         artifact_dir = artifact.download()
#         print(artifact_dir)
#         train_df = pd.read_csv(os.path.join(artifact_dir,'train.csv'),index_col='id')
#         test_df = pd.read_csv(os.path.join(artifact_dir,'test.csv'),index_col='id')
#         pnas_train_df = pd.read_csv(os.path.join(artifact_dir,'PNAS_train.csv'),index_col='id')
    
#     return train_df, test_df, pnas_train_df

# def load_Leaves_Minus_PNAS_dataset():
#     run = wandb.init(reinit=True)
#     # with run:
#     artifact = run.use_artifact('jrose/uncategorized/Leaves-PNAS:v1', type='dataset')
#     artifact_dir = artifact.download()
#     print(artifact_dir)
#     train_df = pd.read_csv(os.path.join(artifact_dir,'train.csv'),index_col='id')
#     test_df = pd.read_csv(os.path.join(artifact_dir,'test.csv'),index_col='id')
    
#     return train_df, test_df