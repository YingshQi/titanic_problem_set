from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import pandas as pd

def map_sex_column(datasets):
    """
    Maps the 'Sex' column to integers (1 for female, 0 for male) for each dataset in the provided list.
    
    Parameters:
    datasets (list): A list of pandas DataFrames to apply the mapping.
    
    Returns:
    None: The function modifies the datasets in place.
    """
    for dataset in datasets:
        dataset['Sex'] = dataset['Sex'].map({'female': 1, 'male': 0}).astype(int)

def split_features_and_target(train_df, valid_df, predictors, target):
    """
    Splits the provided training and validation datasets into features (X) and target (Y).
    
    Parameters:
    train_df (pandas.DataFrame): The training dataset containing both predictors and the target.
    valid_df (pandas.DataFrame): The validation dataset containing both predictors and the target.
    predictors (list): A list of column names to be used as features (X).
    target (str): The column name to be used as the target (Y).
    
    Returns:
    tuple: Returns four items - train_X, train_Y, valid_X, valid_Y.
    """
    train_X = train_df[predictors]
    train_Y = train_df[target].values
    valid_X = valid_df[predictors]
    valid_Y = valid_df[target].values
    
    return train_X, train_Y, valid_X, valid_Y


def train_and_predict(train_X, train_Y, valid_X, random_state=42, n_estimators=100, criterion="gini"):
    """
    Trains a RandomForestClassifier model on the training data and makes predictions on both the training 
    and validation data.
    
    Parameters:
    train_X (pandas.DataFrame or numpy.ndarray): Features for the training dataset.
    train_Y (pandas.Series or numpy.ndarray): Target values for the training dataset.
    valid_X (pandas.DataFrame or numpy.ndarray): Features for the validation dataset.
    random_state (int, optional): Random seed for the RandomForestClassifier (default is 42).
    n_estimators (int, optional): The number of trees in the forest (default is 100).
    criterion (str, optional): The function to measure the quality of a split ("gini" or "entropy", default is "gini").
    
    Returns:
    tuple: Returns predictions for the training data (`preds_tr`) and validation data (`preds`).
    """
    # Initialize the RandomForestClassifier
    clf = RandomForestClassifier(n_jobs=-1, 
                                 random_state=random_state,
                                 criterion=criterion,
                                 n_estimators=n_estimators,
                                 verbose=False)
    
    # Fit the model on the training data
    clf.fit(train_X, train_Y)
    
    # Make predictions on the training and validation sets
    preds_tr = clf.predict(train_X)
    preds = clf.predict(valid_X)
    
    return preds_tr, preds

def generate_classification_report(train_Y, preds_tr, valid_Y, preds):
    """
    Generates and prints classification reports for both training and validation data.
    
    Parameters:
    train_Y (array-like): True target values for the training data.
    preds_tr (array-like): Predicted values for the training data.
    valid_Y (array-like): True target values for the validation data.
    preds (array-like): Predicted values for the validation data.
    
    Returns:
    None
    """
    # Classification report for training data
    print(metrics.classification_report(train_Y, preds_tr, target_names=['Not Survived', 'Survived']))
    
    # Classification report for validation data.
    print(metrics.classification_report(valid_Y, preds, target_names=['Not Survived', 'Survived']))