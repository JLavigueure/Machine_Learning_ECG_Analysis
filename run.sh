# This script is used to run the entire pipeline for data processing, EDA, model fitting, and model evaluation.

# Load raw data at 100hz
python load_data.py data/ptb-xl 100
# Process the ECG signals
python signal_preprocess.py 100
# Perform exploratory analysis and save results to output directory
python exploratory_analysis.py output
# Clean and reduce the data. Split into train and test sets
python clean_data.py
# Resample the datasets to be balanced for each label
python resample_data.py
# Fit models
python fit_models.py accuracy
# Fit voting classifiers using a subset of the models
python python fit_voting_classifier.py data/fit_models_accuracy.pkl LogisticRegression KNN SVC RandomForest XGBClassifier
# Fit a meta-learner using a subset of the models
python fit_meta_learner.py data/fit_models_accuracy.pkl LogisticRegression accuracy LogisticRegression KNN GaussianNB SVC RandomForest BaggingClassifier AdaBoostClassifier XGBClassifier
# Evaluate the performance of the models
python evaluate_models.py data/fit_models_accuracy.pkl

