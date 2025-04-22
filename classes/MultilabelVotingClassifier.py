"""
This script contains the MultilabelVotingClassifier class, which is a custom implementation 
of a multilabel voting classifier using the scikit-learn library. It combines multiple classifiers
into a single model that can predict multiple labels for each instance in the dataset.
Note this class requires the classifiers to be pre-fit before being passed to the constructor.
"""

import pandas as pd
import numpy as np

class MultilabelVotingClassifier:
    def __init__(self, classifiers: dict, voting: str='hard'):
        """
        Initialize the MultilabelVotingClassifier. 

        Parameters:
        -----------
        classifiers : dict
            A dictionary of pre-fit classifiers to be used in the ensemble.
        voting : str, optional
            The type of voting to use. Can be 'soft' or 'hard'.
        """
        self.classifiers_ = classifiers
        self.voting_ = voting

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
        if self.voting_ == 'hard':
            return self.predict_hard_(X)
        elif self.voting_ == 'soft':
            return self.predict_soft_(X)
        else:
            raise ValueError("Voting type must be 'hard' or 'soft'")
        
    def predict_hard_(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict the labels for the given input data using hard voting.

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
        y_pred = np.array(y_pred)
        
        # Use majority voting to determine final predictions
        y_pred = np.mean(y_pred, axis=0) >= 0.5

        return y_pred.astype(int)
    
    def predict_soft_(self, X: pd.DataFrame) -> np.ndarray:
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
            y_pred_proba.append(classifier.predict_proba(X))

        y_pred_proba = np.array(y_pred_proba)
        
        # Average the probabilities across classifiers
        y_pred_proba = np.mean(y_pred_proba, axis=0)

        # Convert probabilities to binary predictions and return
        return (y_pred_proba >= 0.5).astype(int)