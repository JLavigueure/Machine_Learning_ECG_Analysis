"""
This script fits new models using the same parameters as the fitted models but on a subset of the training data, it then fits a meta-learner on the new fitted models 
using the remaining data. It fits two meta-learners: one using the predict method and one using the predict_proba method.

Usage:
    python fit_meta_learner.py <fit_models_path> <meta_learner_base_model> <metric_to_optimize> <ModelName_1> <ModelName_2> ... <ModelName_n>
    Note: If no model names are provided, fit all models.
"""

import pickle
import pandas as pd
import numpy as np
import sys
import os
from sklearn.model_selection import train_test_split
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from classes.MultilabelClassifier import MultilabelClassifier
from classes.MultilabelMetaLearner import MultilabelMetaLearner
from sklearn.model_selection import RandomizedSearchCV
import warnings
import time

def main():
    # Check if cli argument is provided
    if len(sys.argv) < 4:
        print("Usage: python fit_meta_learner.py <fit_models_path> <meta_learner_base_model> <ModelName_1> <ModelName_2> ... <ModelName_n>")
        sys.exit(1)
    
    # Get CLI args
    path = sys.argv[1]
    model_name_arg = sys.argv[2]
    metric = sys.argv[3]
    cli_args = sys.argv[4:]
    random_state = 61297

    # Load the fitted models
    with open(path, 'rb') as f:
        fit_models = pickle.load(f)

    # Load the resampled data
    with open('data/resampled_data.pkl', 'rb') as f:
        resampled_data = pickle.load(f)

    # Create meta-learner base model
    match model_name_arg:
        case 'SVC':
            meta_learner_base_model = SVC(probability=True, random_state=random_state)
        case 'LogisticRegression':
            meta_learner_base_model = LogisticRegression(random_state=random_state)
        case 'KNN':
            meta_learner_base_model = KNeighborsClassifier()
        case 'GaussianNB':
            meta_learner_base_model = GaussianNB()
        case 'RandomForest':
            meta_learner_base_model = RandomForestClassifier(random_state=random_state)
        case 'BaggingClassifier':
            meta_learner_base_model = BaggingClassifier(random_state=random_state)
        case 'AdaBoostClassifier':
            meta_learner_base_model = AdaBoostClassifier(random_state=random_state)
        case 'XGBClassifier':
            meta_learner_base_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state)
        case _:
            raise ValueError(f"Unknown meta-learner base model: {meta_learner_base_model}")
    
    # Parameter grids for each model for random search
    meta_param_grids = {
        'LogisticRegression': {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization strength
        },
        'KNN': {
            'n_neighbors': [5, 10, 20, 50, 100],  # Number of neighbors
            'weights': ['uniform', 'distance'],  # Weighting strategy for neighbors
        },
        'GaussianNB': {
            # No hyperparameters to tune for GaussianNB
        },
        'SVC': {
            'C': [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
            'kernel': ['rbf', 'linear', 'poly'],  # Kernel type
            'gamma': ['scale', 0.001, 0.01, 0.1, 1],  # Kernel coefficient
            'class_weight': ['balanced'],  # Class weight
        },
        'RandomForest': {
            'n_estimators': [50, 100, 200, 500],  # Number of trees in the forest
            'max_depth': [None, 5, 10, 20],  # Max depth of the trees
            'min_samples_split': [2, 5, 10],  # Minimum samples required to split a node
            'max_features': [0.2, 0.4, 0.6, 0.8, 1.0],  # Number of features to consider when looking for the best split
            'bootstrap': [True, False],  # Whether bootstrap samples are used when building trees
        },
        'BaggingClassifier': {
            'estimator': [
                DecisionTreeClassifier(max_depth=3),                                           
                DecisionTreeClassifier(max_depth=5),                        
                DecisionTreeClassifier(max_depth=10),                        
            ],  # Base estimator
            'n_estimators': [10, 50, 100],  # Number of base estimators
            'max_samples': [0.5, 0.7, 1.0],  # Proportion of samples to draw for each base estimator
            'max_features': [0.25, 0.5, 0.75, 1.0],  # Proportion of features to draw for each base estimator
            'bootstrap': [True, False],  # Whether to sample with replacement
        },
        'AdaBoostClassifier': {
            'n_estimators': [50, 100, 200],  # Number of estimators (trees)
            'learning_rate': [0.01, 0.1, 0.5, 1.0],  # Learning rate
        },
        'XGBClassifier': {
            'n_estimators': [100, 200, 500],  # Number of boosting stages (trees)
            'learning_rate': [0.01, 0.1, 0.3],  # Learning rate
            'gamma': [0, 1, 5],  # Min loss reduction required to make a split
            'reg_alpha': [0, 1, 10],  # L1 regularization
            'reg_lambda': [1, 5, 10],  # L2 regularization
            'max_depth': [3, 5, 7, 10],  # Maximum depth of the trees
            'subsample': [0.5, 0.7, 1.0],  # Fraction of samples used for fitting each trees
            'n_jobs': [-1],  # Use all available cores
        }
    }

    random_search_args = {
        'param_distributions': meta_param_grids[model_name_arg],
        'n_iter': 10,
        'scoring': metric, 
        'verbose': 1,
        'cv': 5,
        'random_state': random_state,
        'n_jobs': -1
    }

    # Suppress warnings from randomized search
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn.model_selection._search')   
    
    # Split the resampled data for the base classifiers and meta learner
    for label, (X, y) in resampled_data.items():
        resampled_data[label] = train_test_split(resampled_data[label][0], resampled_data[label][1], test_size=0.20, random_state=random_state) # X_base, X_meta, y_base, y_meta

    # list to store the fitted models
    models = {}

    # Check if the meta-learner base models are already fitted
    fit_base_models_path = 'data/fit_meta_learner_base_models.pkl'
    if os.path.exists(fit_base_models_path):
        with open(fit_base_models_path, 'rb') as f:
            fit_base_models = pickle.load(f)
    else: 
        fit_base_models = {}

    # Refit all models on the subset of the training data
    print('Fitting base models...')
    for model_name, model in fit_models.items():
        # Check if the model is in the cli arguments
        if model_name not in cli_args and cli_args or model_name == 'MultilabelMetaLearner':
            continue
        # Check if model was already fitted
        if model_name in fit_base_models:
            print(f'loaded {model_name} from previously fitted models')
            models[model_name] = fit_base_models[model_name]
            continue
        print(f'Fitting {model_name}...')    
        # Loop through each model within the MultilabelClassifier
        classifiers = {}
        for label, classifier in model.classifiers_.items():
            print(f'Fitting to {label}')
            # Create new instances of the classifiers with the same parameters
            classifiers[label] = (clone(classifier))
            # Fit the new classifiers on the subset of the training data
            classifiers[label].fit(resampled_data[label][0], resampled_data[label][2]) 
        # Create a new MultilabelClassifier instance with the fitted classifiers
        models[model_name] = MultilabelClassifier(classifiers)

    # Save the fitted base models to a pickle file
    with open(fit_base_models_path, 'wb') as f:
        fit_base_models.update(models)
        pickle.dump(fit_base_models, f)

    # Fit meta learners on the fitted models using the remaining data
    print(f'\nFitting {model_name_arg} meta-learners...')
        
    # For each label, extract the predictions and fit a meta-learner
    meta_learners_predict = {}
    meta_learners_proba = {}
    for index, label in enumerate(sorted(resampled_data.keys())): # Sorted to match binarized label order
        X_base, X_meta, y_base, y_meta = resampled_data[label]
        # Get the predictions for the current label
        print(f'\nGathering predictions for {label}...')
        base_predict = []
        base_proba = []
        for model_name, model in models.items():
            base_predict.append(model.predict(X_meta))
            base_proba.append(model.predict_proba(X_meta))
        base_predict = np.array(base_predict)
        base_proba = np.array(base_proba)
        X_predict = pd.DataFrame(base_predict[:,:, index].copy().transpose(), columns=models.keys())
        X_proba = pd.DataFrame(base_proba[:,:, index].copy().transpose(), columns=models.keys())
        
        # Fit the meta-learners using random search for hyperparameter tuning
        print(f'fitting predict meta-learner for {label}')
        time_start = time.time() # Start timer for runtime calculation
        search_predict = RandomizedSearchCV(estimator=clone(meta_learner_base_model), **random_search_args)
        search_predict.fit(X_predict, y_meta)
        print(f"Best parameters: {search_predict.best_params_}")
        print(f"Accuracy: {search_predict.best_score_ * 100:.2f}%")
        print(f'Run time: {time.time() - time_start} seconds')

        print(f'fitting proba meta-learner for {label}')
        time_start = time.time() # Start timer for runtime calculation
        search_proba = RandomizedSearchCV(estimator=clone(meta_learner_base_model), **random_search_args)
        search_proba.fit(X_proba, y_meta)
        print(f"Best parameters: {search_proba.best_params_}")
        print(f"Accuracy: {search_proba.best_score_ * 100:.2f}%")
        print(f'Run time: {time.time() - time_start} seconds')

        # Store the fitted meta-learners for each label
        meta_learners_predict[label] = search_predict.best_estimator_
        meta_learners_proba[label] = search_proba.best_estimator_

    # Combine the meta-learners into a MultilabelMetaLearner
    fit_models[f'{model_name_arg}PredictMultilabelMetaLearner'] = MultilabelMetaLearner(models, meta_learners_predict, stack_method='predict')
    fit_models[f'{model_name_arg}ProbaMultilabelMetaLearner'] = MultilabelMetaLearner(models, meta_learners_proba, stack_method='predict_proba')

    # Save the fitted models to a pickle file
    print(path)
    with open(path, 'wb') as f:
        pickle.dump(fit_models, f)

if __name__ == "__main__":
    main()
