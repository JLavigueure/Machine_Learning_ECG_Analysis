"""
This script contains the MultilabelMetaLearner class, which is a custom implementation of a multilabel meta-learner using the scikit-learn library.
It combines multiple classifiers into a single model that can predict multiple labels for each instance in the dataset.
It uses a meta-learner to combine the predictions of the base classifiers and make final predictions.
Note this class requires the classifiers to be pre-fit before being passed to the constructor.
"""
import pandas as pd
import numpy as np

class MultilabelMetaLearner:
    def __init__(self, base_classifiers: dict, meta_classifiers: dict, stack_method: str='predict'):
        """
        Initialize the MultilabelMetaLearner. 

        Parameters:
        -----------
        classifiers : dict
            A dictionary of prefit base-classifiers to be used in the ensemble.
        meta_classifier : dict
            A dictionary of prefit meta-classifiers to be used in the ensemble. Label names are used as keys.
        stack_method : str, optional
            The method to use for stacking the predictions. Can be 'predict' or 'predict_proba'.
        """
        self.base_classifiers_ = base_classifiers
        self.classifiers_ = meta_classifiers
        self.stack_method_ = stack_method

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
        base_y_pred = []
        for classifier_name in sorted(self.base_classifiers_.keys()):
            classifier = self.base_classifiers_[classifier_name]
            if self.stack_method_ == 'predict_proba':
                base_y_pred.append(classifier.predict_proba(X))
            elif self.stack_method_ == 'predict':
                base_y_pred.append(classifier.predict(X))
            else:
                raise ValueError("Stacking method must be 'predict' or 'predict_proba'")
        base_y_pred = np.array(base_y_pred)
        
        # Feed the predictions to the meta-classifier
        y_pred = []
        for index, (label, meta_classifier) in enumerate(self.classifiers_.items()):
            X = pd.DataFrame(base_y_pred[:, :, index].copy().transpose(), columns=self.base_classifiers_.keys())
            y_pred.append(meta_classifier.predict(X))

        return np.array(y_pred).transpose()