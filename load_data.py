""""
Load data from PTB-XL dataset and store in raw_data.pkl. Data is either sampled at 100Hz or 500Hz.
Data obtained from: https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset/data

Usage: 
    python load_data.py <path_to_dataset_folder> <sampling_rate>
""" 

import os       # for file operations
import sys      # for command line arguments
import ast      # for string to dictionary conversion
import wfdb     # for reading signal data
import pickle   # for saving data
import numpy as np 
import pandas as pd

def load_metadata(path: str) -> pd.DataFrame:
    """
    Load metadata from the PTB-XL dataset.

    Parameters
    ----------
    path : str
        Path to the dataset.
    
    Returns
    -------
    data : DataFrame
        DataFrame containing metadata of the dataset.
    """
    data = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')
    data.scp_codes = data.scp_codes.apply(lambda x: ast.literal_eval(x))
    return data

def load_raw_ecg_data(df: pd.DataFrame, sampling_rate: int, path: str) -> np.array:
    """
    Given a dataframe of the datasets metadata, loop through all correspdoning ECG files
    at the given sampling_rate and load them into a 3D array; one page per entry, one 
    row per signal measurement, one column per ecg lead.
    
    Parameters
    ----------
    df : DataFrame
        DataFrame containing metadata of the dataset.
    sampling_rate : int
        Sampling rate to use for the data.
    path : str
        Path to the data files.

    Returns
    -------
    data : np.array
        3D array containing the ECG data
    """
    if sampling_rate == 100:
        data = [wfdb.rdsamp(path+f) for f in df.filename_lr]
    else:
        data = [wfdb.rdsamp(path+f) for f in df.filename_hr]
    data = np.array([signal for signal, meta in data])
    return data

def append_raw_ecg_data(data_csv: pd.DataFrame, data_ecg: np.array) -> pd.DataFrame:
    """
    Append the raw ECG data to the metadata DataFrame.

    Parameters
    ----------
    data_csv : DataFrame
        DataFrame containing metadata of the dataset.
    data_ecg : np.array
        3D array containing the ECG data.

    Returns
    -------
    data : DataFrame
        DataFrame containing metadata and ECG data.
    """
    ecg_list = [] # list to store ecg data for each patient

    for patient in range(len(data_csv)):    
        curr_pt = [] # list to store ecg data for current patient
        for ecg_lead in range(12):
            # put all ECG readings into current patient list
            curr_pt.append(data_ecg[patient, :, ecg_lead])
        # append patient to overall ecg list
        ecg_list.append(curr_pt)

    # add new column with ecg data for each patient
    data = data_csv.copy()
    data['ecg'] = ecg_list
    return data

def append_diagnostic_superclass(data: pd.DataFrame, path: str) -> pd.DataFrame:
    """
    Append the diagnostic superclass to the metadata DataFrame.

    Parameters
    ----------
    data : DataFrame
        DataFrame containing metadata and ECG data.
    path : str
        Path to the dataset.
    """
    # Load scp_statements.csv for diagnostic aggregation
    agg_df = pd.read_csv(path+'scp_statements.csv', index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic: pd.Series) -> list:
        """
        Given the dictionary of diagnoses for any row, obtain and
        return a set of all their diagnostic superclasses as 
        specified in the dataset files.

        Parameters
        ----------
        y_dic : dict
            Dictionary of diagnoses.
        
        Returns
        -------
        list
            List of diagnostic superclasses.
        """
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    # For each list of diagnoses in scp_codes column, get diagnostic super classes and isnert them into a new column 
    data['diagnostic_superclass'] = data.scp_codes.apply(aggregate_diagnostic)
    return data


def main():
    # Check if correct number of arguments are passed
    if len(sys.argv) != 3:
        print('Usage: python load_data.py <path_to_dataset_folder> <sampling_rate>')
        return

    # path to data and sampling rate of data to use
    path = sys.argv[1]
    sampling_rate = int(sys.argv[2])

    # check if path exists
    if not os.path.exists(path):
        print(f'Path {path} does not exist.')
        return

    # check if sampling rate is valid
    if sampling_rate not in [100, 500]:
        print(f'Invalid sampling rate.')
        return

    # load and convert csv data
    data_csv = load_metadata(path)
    
    # append diagnostic superclass to metadata
    data_csv = append_diagnostic_superclass(data_csv, path)

    # load raw ECG data
    data_ecg = load_raw_ecg_data(data_csv, sampling_rate, path)

    # append raw ECG data to metadata
    data = append_raw_ecg_data(data_csv, data_ecg)

    # save data to pickle file
    if not os.path.exists('data'):
        os.makedirs('data')
    with open('data/raw_data.pkl', 'wb') as f:
        pickle.dump(data, f)

if __name__ == '__main__':
    main()
    

