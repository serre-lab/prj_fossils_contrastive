
import os
import pandas as pd
from typing import List, Dict

def generate_csv_dataset(data_df: pd.DataFrame,
                                      output_path: str,
                                      image_path_col: str='processed_path',
                                      label_col: str='family',
                                      extra_cols: List[str]=None) -> pd.DataFrame:
    """ Helper function for pre-processing a complex DataFrame into a simpler table and writing to an on-disk csv file for efficient data loading later on
    
    Purpose:
        - Discard redundant/irrelevant columns for efficiency at runtime & reducing cognitive load
        - Generate csv artifacts for efforts towards replicability of results
        
    Args:
        data_df: pd.DataFrame
            Complex dataframe meant for simplification and writing to disk
        output_path: str
            File path to which function will write csv file
        image_path_col: str='processed_path'
            Column to keep that contains file paths pointing to images for loading
        label_col: str='family'
            Column to keep for encoding into labels later on. If multiple labels are desired, add the rest in extra_cols.
        extra_cols: List[str]=None
            Any additional columns for which user may anticipate future needs (e.g ID columns like catalog_number)
    
    
    Returns:
        output_df: pd.DataFrame
    
    """
    if extra_cols is None:
        extra_cols = []
    else:
        assert isinstance(extra_cols, list)
    
    output_df = data_df[[image_path_col, label_col, *extra_cols]]
    
    
    output_df.to_csv(output_path)
    assert os.path.isfile(output_path)
    
    return output_df
    



# if __name__ == "__main__":

    # csv_datasets_dir = Path("/media/data_cifs/projects/prj_fossils/data/processed_data/leavesdb-v0_3/catalog_files")
        

    # wilf_csv_path = csv_datasets_dir / "wilf_family_catalog.csv"
    # wilf_df = generate_csv_dataset(data_df=wilf_catalog,
    #                                 output_path=wilf_csv_path,
    #                                 image_path_col='processed_path',
    #                                 label_col='family',
    #                                 extra_cols=['catalog_number'])


    # florissant_csv_path = csv_datasets_dir / "florissant_family_catalog.csv"
    # florissant_df = generate_csv_dataset(data_df=florissant_catalog,
    #                                     output_path=florissant_csv_path,
    #                                     image_path_col='processed_path',
    #                                     label_col='family',
    #                                     extra_cols=['catalog_number'])


    # extant_csv_path = csv_datasets_dir / "extant_family_catalog.csv"
    # extant_df = generate_csv_dataset(data_df=extant_catalog.rename(columns={'updated_family':'family'}),
    #                                 output_path=extant_csv_path,
    #                                 image_path_col='processed_path',
    #                                 label_col='family',
    #                                 extra_cols=['catalog_number'])