"""
This script aggregate the fitted models into a voting classifier and appends the voting classifier to the fitted models pickle file.

Usage:
    python fit_voting_classifier.py <fit_models_path> <ModelName_1> <ModelName_2> ... <ModelName_n>
    Note: If no model names are provided, fit all models.
"""

import pickle
import pandas as pd
import numpy as np
import sys
import os
from classes.MultilabelVotingClassifier import MultilabelVotingClassifier
from classes.MultilabelClassifier import MultilabelClassifier

def main():
    # Get cli arguments
    if len(sys.argv) < 2:
        print("Usage: python fit_voting_classifier.py <fit_models_path> <ModelName_1> <ModelName_2> ... <ModelName_n>")
        sys.exit(1)
    path = sys.argv[1]
    cli_args = sys.argv[2:]
    
    # load the fit models
    with open(path, 'rb') as f:
        fit_models = pickle.load(f)
    
    # list to store the fitted models
    models = {}

    for model_name, model in fit_models.items():
        # Check if the model is in the cli arguments
        if model_name not in cli_args and cli_args or model_name in ['SoftVotingClassifier', 'HardVotingClassifier']:
            continue
        print(f'Including {model_name}')
        models[model_name] = model
    
    print('Creating soft voting classifier...')
    fit_models['SoftVotingClassifier'] = MultilabelVotingClassifier(models, voting='soft')
    print('Creating hard voting classifier...')
    fit_models['HardVotingClassifier'] = MultilabelVotingClassifier(models, voting='hard')

    # Save the fitted models to a pickle file
    print(path)
    with open(path, 'wb') as f:
        pickle.dump(fit_models, f)

if __name__ == "__main__":
    main()