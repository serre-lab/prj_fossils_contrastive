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
from typing import Dict, Union
from boltons.dictutils import OneToOne
import os
import pandas as pd
from pathlib import Path
import wandb

from contrastive_learning.utils.label_utils import ClassLabelEncoder, load_class_labels, save_class_labels


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
