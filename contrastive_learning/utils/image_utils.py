"""
image_utils.py

Created by: Jacob A Rose
Created On: Wednesday, March 18th, 2021

Contains:

func _clever_crop(img, input_size=(128,128), grayscale=False) -> tf.Tensor
func clever_crop(input_size: Tuple[int]=(128,128), grayscale: bool=False) -> Callable:
"""

from collections.abc import Callable
from functools import partial
from typing import Tuple
import tensorflow as tf

from contrastive_learning.utils.label_utils import ClassLabelEncoder, load_class_labels, save_class_labels


def _clever_crop(img: tf.Tensor,
                 target_size: Tuple[int]=(128,128),
                 grayscale: bool=False
                 ) -> tf.Tensor:
    """[summary]

    Args:
        img (tf.Tensor): [description]
        target_size (Tuple[int], optional): [description]. Defaults to (128,128).
        grayscale (bool, optional): [description]. Defaults to False.

    Returns:
        tf.Tensor: [description]
    """    
    maxside = tf.math.maximum(tf.shape(img)[0],tf.shape(img)[1])
    minside = tf.math.minimum(tf.shape(img)[0],tf.shape(img)[1])
    new_img = img
             
    if tf.math.divide(maxside,minside) > 1.2:
        repeating = tf.math.floor(tf.math.divide(maxside,minside))  
        new_img = img
        if tf.math.equal(tf.shape(img)[1],minside):
            for _ in range(int(repeating)):
                new_img = tf.concat((new_img, img), axis=1) 

        if tf.math.equal(tf.shape(img)[0],minside):
            for _ in range(int(repeating)):
                new_img = tf.concat((new_img, img), axis=0)
            new_img = tf.image.rot90(new_img) 
    else:
        new_img = img      
    img = tf.image.resize(new_img, target_size)
    if grayscale:
        img = tf.image.rgb_to_grayscale(img)
        img = tf.image.grayscale_to_rgb(img)
        
    return img

    # img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
    # img = img * 2 - 1

def clever_crop(target_size: Tuple[int]=(128,128),
                grayscale: bool=False
                ) -> Callable:
    """
    Decorates the _clever_crop function with optional kwargs and returns new function that only expects a Tensor as input.
    The returned function can then be mapped onto a tf.data.Dataset

    Args:
        target_size (Tuple[int], optional): [description]. Defaults to (128,128).
        grayscale (bool, optional): [description]. Defaults to False.

    Returns:
        Callable: [description]
    """
    return partial(_clever_crop, target_size=target_size, grayscale=grayscale)