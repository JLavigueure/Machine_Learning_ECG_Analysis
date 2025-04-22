"""
This script contains the MultilabelClassifier class, which is a custom implementation of 
a multilabel classifier using the scikit-learn library. It combines multiple classifiers, 
one per label, into a single model that can predict multiple labels for each instance in the dataset.
Note this class requires the classifiers to be pre-fit before being passed to the constructor.
"""
import pandas as pd
import numpy as np

class MultilabelClassifier:
    def __init__(self, classifiers: dict):
        """
        Initialize the MultilabelClassifier. 
        
        Parameters:
        -----------
        classifiers : dict
            A dictionary of pre-fit classifiers to be used in the ensemble. Label names are used as keys.
        voting : str, optional
            The type of voting to use. Can be 'soft' or 'hard'.
        """
        self.classifiers_ = classifiers
        self.labels_ = list(classifiers.keys())

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the labels for the given input data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            A DataFrame of input data to predict labels for.

        Returns:
        --------
        y_pred : np.ndarray
            A numpy array of predicted labels for each instance in the input data.
        """
        # Gather predictions from each classifier
        y_pred = []
        for classifier_name, classifier in self.classifiers_.items():
            y_pred.append(classifier.predict(X))

        return np.array(y_pred).transpose()
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the probabilities for the given input data.
        
        Parameters:
        -----------
        X : pd.DataFrame
            A DataFrame of input data to predict labels for.

        Returns:
        --------
        y_pred_proba : np.ndarray
            A numpy array of predicted probabilities for each instance in the input data.
        """
        # Gather probabilities from each classifier
        y_pred_proba = []
        for classifier_name, classifier in self.classifiers_.items():
            classifier_pred = classifier.predict_proba(X)
            y_pred_proba.append(classifier_pred[:, 1])  # Get the probability of the positive class

        return np.array(y_pred_proba).transpose()