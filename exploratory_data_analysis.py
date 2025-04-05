""""
Perform exploratory data analysis on the uncleaned processed ECG data and metadata.

Usage: 
    python exploratory_data_analysis.py <optional_output_directory_w/out_slash>
""" 

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle       # for loading data
import os           # for file operations
import sys          # for command line arguments

def csv_missing_values(data: pd.DataFrame, path: str='missing_values') -> None:
    """
    Save count of missing values per column to csv.

    Parameters
    ----------
    data_exploded : pd.Dataframe
        Data.
    path : str, optional
        Directory path to save the CSV file. Default is ''.
    """
    directory = f'{output_directory}{path}'
    if path: os.makedirs(f'{directory}', exist_ok=True)
    path = f'{directory}/MissingValues.csv'
    print(path)
    missing_values = data.isna().sum().sort_values(ascending=False)
    missing_values.to_csv(path, header=False)

def csv_missing_values_group_by_target(data_exploded: pd.DataFrame, path: str='missing_values') -> None:
    """
    Save count of missing values per column and grouped by diagnoses to csv.

    Parameters
    ----------
    data_exploded : pd.Dataframe
        Data.
    path : str, optional
        Directory path to save the CSV file. Default is ''.
    """
    directory = f'{output_directory}{path}'
    if path: os.makedirs(directory, exist_ok=True)
    path = f'{directory}/MissingValuesGroupedByTarget.csv'
    print(path)
    missing_values = data_exploded.groupby('diagnostic_superclass').apply(lambda x: x.isna().sum(), include_groups=False)
    missing_values.to_csv(path)

def plot_target_distribution(data_exploded: pd.DataFrame, path: str='distributions') -> None:
    """
    Save plot of target distribution.

    Parameters
    ----------
    data_exploded : pd.Dataframe
        Data.
    path : str, optional
        Directory path to save the plot. Default is ''.
    """
    directory = f'{output_directory}{path}'
    if path: os.makedirs(directory, exist_ok=True)
    path = f'{directory}/TargetDistribution.png'
    print(path)
    plt.figure(1)
    data_exploded['diagnostic_superclass'].value_counts().plot(kind='bar')
    plt.title('Distribution of diagnoses')
    plt.ylabel('Frequency')
    plt.xlabel('Diagnoses Category')
    plt.grid()
    plt.savefig(path)
    plt.close()

def plot_feature_distribution(data_exploded: pd.DataFrame, column: str, bins: list=None, path: str='distributions') -> None:
    """
    Save plot of feature distribution in relation to target classes.

    Parameters
    ----------
    data_exploded : pd.Dataframe
        Data.
    column : str
        Feature column to analyze.
    bins : list, optional
        Bins for numeric features. Default is None.
    path : str, optional
        Directory path to save the plot. Default is ''.
    """
    directory = f'{output_directory}{path}'
    if path: os.makedirs(directory, exist_ok=True)
    path = f'{directory}/{column.title()}Distribution.png'
    print(path)
    data_exploded = data_exploded.copy()
    if bins is not None:
        labels = [f'{val}-{bins[index+1]}' for index, val in enumerate(bins[:-1])]
        group_col =f'{column}_group'
        data_exploded[group_col] = pd.cut(data_exploded[column], bins=bins, labels=labels, right=False)
    else:
        group_col = column
    counts = data_exploded.groupby([group_col, 'diagnostic_superclass'], observed=True).size().unstack(fill_value=0)
    counts.plot(kind='bar', figsize=(10, 3))
    plt.title(f'Distribution of {column} by Target')
    plt.ylabel('Frequency')
    plt.xlabel(column)
    plt.legend(title='Diagnoses')
    plt.savefig(path)
    plt.close()

def csv_correlations_to_target(data_exploded: pd.DataFrame, path: str='correlations') -> None:
    """
    Save correlation of all features to each target class in respective CSVs.

    Parameters
    ----------
    data_exploded : pd.Dataframe
        Data.
    path : str, optional
        Directory path to save the CSV files. Default is ''.
    """
    directory = f'{output_directory}{path}'
    if path: os.makedirs(directory, exist_ok=True)
    data_exploded = data_exploded.copy()
    for target_class in data_exploded['diagnostic_superclass'].unique():
        path = f'{directory}/FeatureCorrelationTo{target_class}.csv'
        print(path)
        if pd.isna(target_class): continue
        data_exploded['target'] = data_exploded['diagnostic_superclass'].apply(lambda x: 1 if target_class == x else 0)
        corr_matrix = data_exploded.drop(columns=['diagnostic_superclass']).corr()
        corr_matrix = corr_matrix['target'].sort_values(ascending=False)
        corr_matrix.to_csv(path, header=False)

def plot_correlation_to_target(data_exploded: pd.DataFrame, column: str, bins: int=70, path: str='correlations') -> None:
    """
    Save a plot for a feature and its correlation to all the targets.

    Parameters
    ----------
    column : str
        Feature column to analyze.
    bins : int, optional
        Number of bins for numeric features. Default is 70.
    path : str, optional
        Directory path to save the plot. Default is ''.
    """
    directory = f'{output_directory}{path}'
    if path: os.makedirs(directory, exist_ok=True)
    path = f'{directory}/{column}CorrelationToTarget.png'
    print(path)
    data_exploded = data_exploded.copy()
    target = 'diagnostic_superclass'

    if pd.api.types.is_numeric_dtype(data_exploded[column]):
        # Bin continuous data
        data_exploded[f'{column}_binned'] = pd.cut(data_exploded[column], bins=70)
        grouped_data = data_exploded.groupby([f'{column}_binned', 'diagnostic_superclass'], observed=True).size().reset_index(name='count')
        grouped_data[column] = grouped_data[f'{column}_binned'].apply(lambda x: x.mid)
    else:
        # Directly group categorical data
        grouped_data = data_exploded.groupby([column, target], observed=True).size().reset_index(name='count')
    
    plt.figure(figsize=(10, 3))
    for target_class in grouped_data[target].unique():
        target_data = grouped_data[grouped_data[target] == target_class]
        plt.scatter(target_data[column], [target_class] * len(target_data),
                    alpha=0.7, s=target_data['count'] * 5, c=target_data['count'], cmap='plasma')
    
    plt.title(f'Correlation of {column} and {target}')
    plt.xlabel(column)
    plt.ylabel('Target Class')
    plt.colorbar(label='Frequency')
    plt.savefig(path)
    plt.close()
        
def plot_correlation_matrix(data_exploded: pd.DataFrame, path: str='correlations') -> None:
    """
    Create a correlation matrix for the dataset. 

    Paramaters
    ----------
    data_exploded : pd.DataFrame
        Data.
    path : str, optional
        Directory path to save the correlation matrices. Default is 'matricies'.
    """
    directory = f'{output_directory}{path}'
    if path: os.makedirs(directory, exist_ok=True)
    path = f'{directory}/CorrelationMatrix.png'
    print(path)
    data_exploded = pd.get_dummies(data_exploded, columns=['diagnostic_superclass'], drop_first=True)
    plt.figure(figsize=(100,100))
    sns.heatmap(data_exploded.corr(), cmap='Oranges', square=True, cbar=False)
    plt.savefig(path)

def main():
    # Set output directory for figures
    global output_directory
    output_directory = ''
    if len(sys.argv) > 1: 
        output_directory = sys.argv[1] + '/'
        # if directory doesn't exist, create it
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
    
    # Load the pickled dataframe
    with open('data/processed_data.pkl', 'rb') as f:
        data = pickle.load(f)

    # Remove arbitrary features 
    cols_to_drop = [
        'patient_id', 'nurse', 'site', 'device', 'recording_date', 'report', 'scp_codes', 
        'heart_axis', 'infarction_stadium1', 'infarction_stadium2', 'validated_by', 'second_opinion', 
        'initial_autogenerated_report', 'validated_by_human', 'baseline_drift', 'static_noise', 
        'burst_noise', 'electrodes_problems', 'extra_beats', 'pacemaker', 'filename_lr', 'filename_hr',
        'ecg', 'strat_fold'
    ]
    data.drop(columns=cols_to_drop, inplace=True)

    # Explode multiclass columns to multiple rows
    data_exploded = data.explode('diagnostic_superclass').reset_index()

    # Count missing values
    csv_missing_values(data)
    csv_missing_values_group_by_target(data_exploded)

    # Plot distribution of target and features
    plot_target_distribution(data_exploded)
    # Metadata
    plot_feature_distribution(data_exploded, 'sex')
    plot_feature_distribution(data_exploded, 'age', range(0, 101, 5))
    plot_feature_distribution(data_exploded, 'height', range(100, 211, 5))
    plot_feature_distribution(data_exploded, 'weight', range(10, 150, 5))
    # ECG Features
    for column in data_exploded.columns[7:]:
        col_data = data_exploded[column].dropna()
        # Create bins
        lower_bound, upper_bound = np.percentile(col_data, [0, 100])
        bin_width = max(0.01, (upper_bound - lower_bound) / 15)
        bins = np.arange(lower_bound, upper_bound + bin_width, bin_width)
        plot_feature_distribution(data_exploded, column, bins=bins)

    # Calculate all feature's correlations to target
    csv_correlations_to_target(data_exploded)

    # Plot correlation of each feature to the target
    for column in data_exploded.columns:
        if column == 'diagnostic_superclass': continue
        plot_correlation_to_target(data_exploded, column)

    # Plot correlation Matrix
    plot_correlation_matrix(data_exploded)

    

if __name__ == "__main__":
    main()


    