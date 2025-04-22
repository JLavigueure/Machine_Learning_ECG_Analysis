"""
Fit models to the training data. Use random search with cross-validation to tune hyperparameters.
Save the fitted models as a dictionary in a pickle file.

Usage: 
    python fit_models.py <metric_to_optimize> <ModelName_1> <ModelName_2> ... <ModelName_n>
    Note: If no model names are provided, fit all models.
    Available models: SVC, LogisticRegression, KNN, GaussianNB, RandomForest, BaggingClassifier, AdaBoostClassifier, XGBClassifier
"""

import pandas as pd
import numpy as np
import pickle
import os
# Preprocessing
from sklearn.model_selection import train_test_split
# Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
# Hyperparameter tuning
from sklearn.model_selection import RandomizedSearchCV
# Silence warnings
import warnings
# Custom classes
from classes.MultilabelClassifier import MultilabelClassifier
# Execution time
import time
# CLI arguments
import sys

def main():
    random_state = 61297

    # Check if cli argument is provided
    if len(sys.argv) < 2:
        print("Usage: python fit_models.py <metric_to_optimize> <ModelName_1> <ModelName_2> ... <ModelName_n>")
        sys.exit(1)
    # Get CLI args
    metric = sys.argv[1]
    path = f'data/fit_models_{metric}.pkl'
    cli_args = sys.argv[2:]

    # Load the resampled data
    with open('data/resampled_data.pkl', 'rb') as f:
        resampled_data = pickle.load(f)

    # If fit models already exist, load them and append new models
    if os.path.exists(path):
        with open(path, 'rb') as f:
            fit_models = pickle.load(f)
            if not fit_models:
                fit_models = {}
    # Else create an empty dictionary to store the fitted models
    else:
        fit_models = {}

    # Models to fit
    models = {
        # Individual models
        'SVC': SVC(random_state=random_state, probability=True),
        'LogisticRegression': LogisticRegression(random_state=random_state),
        'KNN': KNeighborsClassifier(),
        'GaussianNB': GaussianNB(),
        # Ensemble models
        'RandomForest': RandomForestClassifier(random_state=random_state),
        'BaggingClassifier': BaggingClassifier(random_state=random_state),
        'AdaBoostClassifier': AdaBoostClassifier(random_state=random_state),
        'XGBClassifier': XGBClassifier(random_state=random_state)
    }

    # Parameter grids for each model for random search
    param_grids = {
        'LogisticRegression': {
            'C': [0.01, 1, 5, 10, 20, 50],  # Regularization strength
        },
        'KNN': {
            'n_neighbors': [300, 400, 500, 600, 700, 800],  # Number of neighbors
            'weights': ['uniform', 'distance'],  # Weighting strategy for neighbors
        },
        'GaussianNB': {
            # No hyperparameters to tune for GaussianNB
        },
        'SVC': {
            'C': [0.001, 0.01, 0.1],  # Regularization strength
            'kernel': ['rbf'],  # Type of kernel
            'gamma': [0.001, 0.01, 0.1, 1],  # Kernel coefficientW
            'class_weight': ['balanced'],  # Class weight
        },
        'RandomForest': {
            'n_estimators': [100, 200, 300, 400, 500],  # Number of trees in the forest
            'max_depth': [2, 5, 10, 15, 20],  # Max depth of the trees
            'min_samples_split': [100, 200, 300, 400, 500],  # Minimum samples required to split a node
            'max_features': [None, 0.2, 0.4, 0.6, 0.8],  # Number of features to consider when looking for the best split
            'bootstrap': [True, False]  # Whether bootstrap samples are used when building trees
        },
        'BaggingClassifier': {
                'estimator': [
                    DecisionTreeClassifier(max_depth=5, min_samples_leaf=200),                                           
                    DecisionTreeClassifier(max_depth=10, min_samples_leaf=200),                        
                    DecisionTreeClassifier(max_depth=15, min_samples_leaf=200),                        
                ], # Base estimator
            'n_estimators': [200, 300, 400],  # Number of base estimators
            'max_samples': [0.5, 0.7, 1],  # Proportion of samples to draw for each base estimator
            'max_features': [0.1, 0.25, 0.5, 0.7],  # Proportion of features to draw for each base estimator
            'bootstrap': [True],  # Whether to sample with replacement
        },
        'AdaBoostClassifier': {
            'n_estimators': [50, 100, 200, 500, 700],  # Number of estimators (trees)
            'learning_rate': [0.01, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0],  # Learning rate
        },
        'XGBClassifier': {
            'n_estimators': [1000],  # Number of boosting stages (trees)
            'learning_rate': [0.1],  # Learning rate
            'gamma': [5, 7, 10],  # Min loss reduction required to make a split
            'reg_alpha': [10, 15, 20, 25],  # L1 regularization
            'reg_lambda': [10, 15, 20, 25],  # L2 regularization
            'max_depth': [3, 5, 7, 9],  # Maximum depth of the trees
            'subsample': [0.2, 0.4, 0.6, 0.8, 1.0],  # Fraction of samples used for fitting each trees
            'n_jobs': [-1],  # Use all available cores
        }
    }

    # Suppress warnings from randomized search
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.model_selection._search')   

    # Verify that all arguments are valid model names
    for model_name in cli_args:
        if model_name not in models:
            print(f"Invalid model name: {model_name}. Available models: {', '.join(models.keys())}")
            sys.exit(1)

    # Iterate over each model 
    for model_name, param_grid in param_grids.items():
        # Skip models not specified in CLI arguments
        # If no CLI arguments are provided, fit all models
        if model_name not in cli_args and cli_args:
            continue    
        print(f"Fitting {model_name}...")

        # Create a dictionary to store current model for each resampled dataset
        ensemble = {}

        # Iterate over each resampled dataset (1 per label)
        for label in sorted(resampled_data.keys()): # Sorted to match binarized label order
            print(f"Fitting to {label}...")
            X_train, y_train = resampled_data[label]
    
            # Start timer
            time_start = time.time()

            # Perform random search with cross-validation
            random_search = RandomizedSearchCV(
                estimator = models[model_name],
                param_distributions = param_grid,
                cv = 5,  # Cross-validation folds
                verbose = 1, # Print progress
                n_jobs = -1, # Use all available cores
                n_iter = 10, # Number of parameter settings to sample
                scoring=metric, # Metric to optimize
                random_state = random_state
            ) 
            random_search.fit(X_train, y_train) 
            ensemble[label] = random_search.best_estimator_
            print(f"Best parameters: {random_search.best_params_}")
            print(f"{metric}: {random_search.best_score_ * 100:.2f}%")
            print(f'Run time: {time.time() - time_start} seconds')
        
        # Combine the ensemble of classifiers into a single model
        fit_models[model_name] = MultilabelClassifier(ensemble)
        
        print() # Add a blank line for readability

    # Save the fitted models to a pickle file
    print(path)
    with open(path, 'wb') as f:
        pickle.dump(fit_models, f)

if __name__ == "__main__":
    main()