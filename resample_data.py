"""
Create a target column for each independent label, then duplicate the dataset and balance each target column
using tomek links and random oversampling. Save the results as a dictionary in a pickle file.

Usage:
    python resample_data.py
"""

import pandas as pd
import numpy as np
import pickle
import os
# Resampling
from imblearn.under_sampling import TomekLinks
from imblearn.over_sampling import RandomOverSampler

def main():
    # Load the cleaned data
    with open('data/cleaned_data.pkl', 'rb') as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    # Create a dictionary to store the resampled datasets
    resampled_datasets = {}

    # Iterate over each unique label in the target column
    for label in y_train.explode().unique():
        if pd.isna(label):
            continue
        print('Resampling ', label)
        # Create a new dataset for each label
        new_data_X = X_train.copy()
        # Create a binary target column
        new_data_y = y_train.apply(lambda x: 1 if label in x else 0)  # Create a binary target column
        # Perform Tomek links and random oversampling
        resampled_datasets[label] = TomekLinks().fit_resample(new_data_X, new_data_y)
        resampled_datasets[label] = RandomOverSampler(random_state=61297).fit_resample(resampled_datasets[label][0], resampled_datasets[label][1])
        
    # Save the resampled datasets to a pickle file
    path = 'data/resampled_data.pkl'
    print(path)
    with open(path, 'wb') as f:
        pickle.dump(resampled_datasets, f)

if __name__ == "__main__":
    main()