"""
DEPRECATED - Wednesday March 17th, 2021 by Jacob Rose
"""


# import os
# import pandas as pd
# import wandb
# import numpy as np
# from typing import Dict, Union
# import tensorflow as tf



# def class_counts(data_df, label_col='family'):
#     return data_df.value_counts(label_col)

# def class_weights(data_df, label_col='family'):
#     counts = class_counts(data_df, label_col='family')
#     total_count = counts.sum()

#     weights = total_count / counts

#     return weights



# class DatasetInfo:
#     csv_dataset_dir: str= '/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files/' 
#     dataset_id: str=None # e.g. "extant_family_10"

#     def __init__(self, label_col='family'):
#         train_df = pd.read_csv(self.csv_dataset_dir + self.dataset_id + '_train.csv')
#         val_df = pd.read_csv(self.csv_dataset_dir + self.dataset_id + '_val.csv')
#         test_df = pd.read_csv(self.csv_dataset_dir + self.dataset_id + '_test.csv')

#         self.label_col = label_col
#         self.class_labels = sorted(set(train_df[label_col].values))
#         self.class_labels_str2int = {label:idx for idx, label in enumerate(self.class_labels)}
#         self.class_labels_int2str = {idx:label for label, idx in self.class_labels_str2int.items()}

#         self.train_df = train_df.assign(label = train_df[label_col].apply(lambda x: self.class_labels_str2int[x]))
#         self.val_df = val_df.assign(label = val_df[label_col].apply(lambda x: self.class_labels_str2int[x]))
#         self.test_df = test_df.assign(label = test_df[label_col].apply(lambda x: self.class_labels_str2int[x]))
#         self.NUM_CLASSES = len(self.class_labels)
#         self.NUM_SAMPLES = {'train':len(self.train_df),
#                             'val':len(self.val_df),
#                             'test':len(self.test_df)}

#     def as_dataframes(self):
#         return self.train_df, self.val_df, self.test_df







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



# def load_train_test_artifact(artifact_uri='jrose/uncategorized/Leaves-PNAS:v1', run=None):
#     if run is None:
#         run = wandb.init(reinit=True, settings=wandb.Settings(start_method="fork"))
    
#     # with run:
#     artifact = run.use_artifact(artifact_uri, type='dataset')
#     artifact_dir = artifact.download()
#     print('artifact_dir =',artifact_dir)
#     train_df = pd.read_csv(os.path.join(artifact_dir,'train.csv'),index_col='id')
#     test_df = pd.read_csv(os.path.join(artifact_dir,'test.csv'),index_col='id')
#     return train_df, test_df




# def load_dataset_from_artifact(dataset_name='Fossil', threshold=4, test_size=0.3, version='latest', artifact_name=None, run=None):
#     train_size = 1 - test_size
#     if artifact_name:
#         pass
#     elif dataset_name=='Fossil':
#         artifact_name = f'{dataset_name}_{threshold}_{int(train_size*100)}-{int(100*test_size)}:{version}'
#     elif dataset_name=='PNAS':
#         artifact_name = f'{dataset_name}_family_{threshold}_50-50:{version}'
#     elif dataset_name=='Leaves':
#         artifact_name = f'{dataset_name}_family_{threshold}_{int(train_size*100)}-{int(100*test_size)}:{version}'
#     elif dataset_name=='Leaves-PNAS':
#         artifact_name = f'{dataset_name}_{int(train_size*100)}-{int(100*test_size)}:{version}'
#     elif dataset_name in ['Leaves_in_PNAS', 'PNAS_in_Leaves']:
#         artifact_name = f'{dataset_name}_{int(train_size*100)}-{int(100*test_size)}:{version}'
            
#     artifact_uri = f'brown-serre-lab/paleoai-project/{artifact_name}'
#     return load_train_test_artifact(artifact_uri=artifact_uri, run=run)



# def _clever_crop(img,input_size = (128,128),grayscale=False):
#     maxside = tf.math.maximum(tf.shape(img)[0],tf.shape(img)[1])
#     minside = tf.math.minimum(tf.shape(img)[0],tf.shape(img)[1])
#     new_img = img
             
#     if tf.math.divide(maxside,minside) > 1.2:
#         repeating = tf.math.floor(tf.math.divide(maxside,minside))  
#         new_img = img
#         if tf.math.equal(tf.shape(img)[1],minside):
#             for i in range(int(repeating)):
#                 new_img = tf.concat((new_img, img), axis=1) 

#         if tf.math.equal(tf.shape(img)[0],minside):
#             for i in range(int(repeating)):
#                 new_img = tf.concat((new_img, img), axis=0)
#             new_img = tf.image.rot90(new_img) 
#     else:
#         new_img = img      
#     img = tf.image.resize(new_img, input_size)
#     if grayscale:
#         img = tf.image.rgb_to_grayscale(img)
#         img = tf.image.grayscale_to_rgb(img)
#         #img = tf.image.random_crop(img, [crop_size, crop_size, tf.shape(img)[-1]])
#         # 
#     # # Move this outside
#     # img = tf.clip_by_value(img, 0, 255) / 255.0  # or img = tl.minmax_norm(img)
#     # img = img * 2 - 1
    
#     return img  