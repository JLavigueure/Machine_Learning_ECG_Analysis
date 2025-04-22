"""
Evaluate the performance of the models on the test set ans save the results to a CSV file.

Usage:
    python evaluate_models.py <fit_models_path> <model_name_1> <model_name_2> ... <model_name_n>
    Note: If no model names are provided, evaluate all models.
"""

import pandas as pd
import numpy as np
import pickle
import os
import sys
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from classes.MultilabelClassifier import MultilabelClassifier
import time

def evaluate_model(y_pred: pd.Series, y_true: pd.Series) -> dict:
    """
    Evaluate the model for a specific label using accuracy, precision, recall, and F1 score.

    Parameters:
    -----------
    y_pred : pd.Series
        The predicted labels for the data.
    y_true : pd.Series
        The true labels for the data.
    label_index : int
        The index of the label to evaluate. See `mlb.classes_` for the mapping.
    
    Returns:
    --------
    metrics : dict
        A dictionary containing the accuracy, precision, recall, F1 score, and AUC.
    """
    # Loop through prediction results to get results for the specific label
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'AUC': roc_auc_score(y_true, y_pred)
    }
    return metrics

def main():
    # CLI arguments
    if len(sys.argv) < 2:
        print("Usage: python evaluate_models.py <fit_models_path> <model_name_1> <model_name_2> ... <model_name_n>")
        sys.exit(1)
    path = sys.argv[1]
    cli_args = sys.argv[2:]

    # Load the cleaned data
    with open('data/cleaned_data.pkl', 'rb') as f:
        X_train, X_test, y_train, y_test = pickle.load(f)

    # Load fit models
    with open(path, 'rb') as f:
        fit_models = pickle.load(f)

    # Binarize the test labels
    mlb = MultiLabelBinarizer()
    y_test = mlb.fit_transform(y_test)

    # Initialize a dataframe to store metrics
    metrics_df = pd.DataFrame(columns=['Model', 'Label', 'Params', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC'])

    # Evaluate each model
    for model_name, model in fit_models.items():
        # Check if the model is in the cli arguments
        if model_name not in cli_args and cli_args:
            continue
        print(f'Evaluating {model_name}...')
        start_time = time.time()
        y_pred = model.predict(X_test) 
        # Evaluate performance for each label
        labels = sorted(y_train.explode().dropna().unique())
        for i in range(len(labels)):
            label = labels[i]
            metrics = evaluate_model(y_pred[:,i], y_test[:,i])
            metrics['Model'] = model_name
            metrics['Label'] = label
            if label in model.classifiers_ and hasattr(model.classifiers_[label], 'get_params'):
                metrics['Params'] = model.classifiers_[label].get_params()
            else:
                metrics['Params'] = 'N/A'
            metrics_df.loc[len(metrics_df)] = metrics
        print(f"Evaluation time: {time.time() - start_time:.2f} seconds")
        print(f"{metrics_df[metrics_df['Model'] == model_name][['Label', 'Accuracy', 'Precision', 'Recall']]}")

    # Add a row for the average of all labels per model
    avg_metrics = metrics_df[['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']].groupby('Model').mean().reset_index()
    avg_metrics['Label'] = 'All'
    avg_metrics['Params'] = 'N/A'
    metrics_df = pd.concat([metrics_df, avg_metrics], ignore_index=True)
    metrics_df = metrics_df.sort_values(by=['Model', 'Label'])

    # Round all numbers in metrics dataframe to 4 decimal places
    metrics_df = metrics_df.round(4)

    # Format all numbers as percentages
    for col in ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']:
        metrics_df[col] = metrics_df[col].apply(lambda x: f'{x:.2%}')

    # Save the metrics dataframe to a CSV file
    os.makedirs('metrics', exist_ok=True)
    path = f'metrics/{path[path.rindex('/')+1:path.rindex('.pkl')]}_metrics.csv'
    metrics_df.to_csv(path, index=False)
    print(f'Metrics saved to {path}')

if __name__ == '__main__':
    main()

