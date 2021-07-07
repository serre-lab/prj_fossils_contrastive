"""
Script for cleaning, formatting, and logging image classification datasets to WandB artifacts, to be uploaded to the cloud for machine-agnostic querying.

Created by: Sunday May 16th, 2021
Author: Jacob A Rose

Example cmdline use:


python "./prj_fossils_contrastive/contrastive_learning/data/pytorch/wandb/create_versioned_dataset_artifacts.py" \
        --dataset_name="Extant_family_10_512" \
        --val_size=0.2                        \
        --label_type="family"                 \
        --image_size=512                      \
        --channels=3                          \
        --seed=389
        
        
python "./prj_fossils_contrastive/contrastive_learning/data/pytorch/wandb/create_versioned_dataset_artifacts.py" \
        --dataset_name="PNAS_family_10_1024"  \
        --val_size=0.2                        \
        --label_type="family"                 \
        --image_size=1024                     \
        --channels=3                          \
        --seed=389
        



"""




from pathlib import Path
from tqdm.auto import tqdm
from torch.utils.data import Dataset
from typing import Dict
from wandb.apis.public import Artifact

import argparse
import wandb
from contrastive_learning.data.pytorch.datamodules import get_datamodule


parser = argparse.ArgumentParser(description='Script for cleaning, formatting, and logging image classification datasets to WandB artifacts, to be uploaded to the cloud for machine-agnostic querying.')

parser.add_argument('--dataset_name', type=str, required=True,
                    help='name for this dataset')

parser.add_argument('--val_size', type=float, default=0.2,
                    help='Fraction of the train set to be used for the validation set')
parser.add_argument('--label_type', type=str, default="family",
                    help='Taxonomic column to use for encoding into int class labels.')
parser.add_argument('--image_size', type=int, default=512,
                    help='integer number of pixels for the height and width of resized images prior to use.')
parser.add_argument('--channels', type=int, default=3,
                    choices=[3, 1],
                    help='Number of desired channels for images to use or be converted to use.')
parser.add_argument('--seed', type=int, default=389,
                    help='random seed')



def main():
    """
    Pass args to script via commandline to run this function.
    
    Produces train, val, and test data subsets and logs them as multimedia Table objects within WandB artifacts to WandB.
    
    """
    config = parser.parse_args()
    config.normalize=False
    
    print(f'Initiating raw-data image dataset creation process using the following configuration:\n{config}')
    
    wandb.init(project="image_classification_datasets",
               job_type='create-dataset',
               config=config)
    config.name = config.dataset_name
    
    dataset_artifact = wandb.Artifact(config.name,# config["dataset_name"],
                                      type='raw_data')

    datamodule = get_datamodule(config)

    datamodule.setup(stage='fit')
    datamodule.setup(stage='test')

    
    register_raw_dataset(dataset=datamodule.train_dataset,
                         artifact=dataset_artifact,
                         subset='train',
                         fix_catalog_number=True)

    del datamodule.train_dataset

    register_raw_dataset(dataset=datamodule.val_dataset,
                         artifact=dataset_artifact,
                         subset='val',
                         fix_catalog_number=True)

    del datamodule.val_dataset

    register_raw_dataset(dataset=datamodule.test_dataset,
                         artifact=dataset_artifact,
                         subset='test',
                         fix_catalog_number=True)

    del datamodule.test_dataset
    
    wandb.log_artifact(dataset_artifact)
    wandb.finish()
    









def get_labels_from_filepath(path: str, fix_catalog_number: bool = False) -> Dict[str,str]:
    """
    Splits a precisely-formatted filename with the expectation that it is constructed with the following fields separated by '_':
    1. family
    2. genus
    3. species
    4. collection
    5. catalog_number
    
    If fix_catalog_number is True, assume that the collection is not included and must separately be extracted from the first part of the catalog number.
    
    """
    family, genus, species, collection, catalog_number = Path(path).stem.split("_", maxsplit=4)
    if fix_catalog_number:
        catalog_number = '_'.join([collection, catalog_number])
    return {"family":family,
            "genus":genus,
            "species":species,
            "collection":collection,
            "catalog_number":catalog_number}

# construct a table containing our dataset
def register_raw_dataset(dataset            : Dataset,
                         artifact           : Artifact,
                         subset             : str=None,
                         fix_catalog_number : bool=False,
                         label_type         : str="family") -> None:
    
    print(f"Registering {len(dataset.samples)} samples in {subset} subset with labels encoded to the {label_type} column.")
    table = wandb.Table(['image',
                         'label', 
                         "family",
                         "genus",
                         "species",
                         "collection",
                         "catalog_number"])
    for sample in tqdm(dataset.samples):
        path, label = sample        
        metadata = get_labels_from_filepath(path, fix_catalog_number=fix_catalog_number)
        rel_path = f"{metadata[label_type]}/{Path(path).name}"
        if isinstance(subset, str):
            rel_path = f'{subset}/' + rel_path
        
        artifact.add_file(path, rel_path)
        table.add_data(wandb.Image(path), label, *list(metadata.values()))
        
        
    if hasattr(dataset, 'image_size'):
        if isinstance(dataset.image_size, int):
            artifact.metadata['image_size'] = dataset.image_size
        
        
    artifact.metadata['num_samples'] = len(dataset.samples)
    artifact.add(table, f'tables/{subset}')
    
    
    
    
if __name__ == "__main__":
    
    
    main()
