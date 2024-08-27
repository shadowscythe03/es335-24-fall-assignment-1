"""
You can add your own functions here according to your decision tree implementation.
There is no restriction on following the below template, these fucntions are here to simply help you.
"""

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    """
    Function to perform one hot encoding on the input data
    """

    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    """
    Function to check if the given series has real or discrete values
    """
    return pd.api.types.is_numeric_dtype(y)
    pass


def entropy(Y: pd.Series) -> float:
    """
    Function to calculate the entropy
    """
    probs = Y.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs+1e-10))


def gini_index(Y: pd.Series) -> float:
    """
    Function to calculate the gini index
    """
    probs = Y.value_counts(normalize=True)
    return 1-np.sum(probs**2)

def mse(y: pd.Series) -> float:
    """Calculate the Mean Squared Error (MSE) of a label distribution."""
    mean = y.mean()
    return np.mean((y - mean) ** 2)

def information_gain(Y: pd.Series, attr: pd.Series, criterion: str) -> float:
    """
    Function to calculate the information gain using criterion (entropy, gini index or MSE)
    """
    # Calculate the impurity of the original dataset
    if criterion == 'information_gain':
        total_impurity = entropy(Y)
    elif criterion == 'gini_index':
        total_impurity = gini_index(Y)
    # elif criterion == 'mse':
    else:
        total_impurity = mse(Y)
    # else:
    #     raise ValueError("Criterion should be 'entropy', 'gini', or 'mse'")
    
    # Calculate the weighted impurity after splitting on the attribute
    weighted_impurity = 0.0
    for value in attr.unique():
        subset_Y = Y[attr == value]
        weight = len(subset_Y) / len(Y)
        
        if criterion == 'information_gain':
            subset_impurity = entropy(subset_Y)
        elif criterion == 'gini_index':
            subset_impurity = gini_index(subset_Y)
        else:
            subset_impurity = mse(subset_Y)
        
        weighted_impurity += weight * subset_impurity
    
    # Information gain is the difference between the total impurity and the weighted impurity
    return total_impurity - weighted_impurity

def best_split_continuous(X: pd.Series, y: pd.Series, criterion: str) -> tuple:
    sorted_X = X.sort_values()
    sorted_y = y.loc[sorted_X.index]
    
    best_gain = -float('inf')
    best_split = None

    for i in range(1, len(sorted_X)):
        if sorted_X.iloc[i] == sorted_X.iloc[i - 1]:
            continue
        split_value = (sorted_X.iloc[i] + sorted_X.iloc[i - 1]) / 2
        
        left_mask = X <= split_value
        right_mask = X > split_value
        
        left_gain = information_gain(sorted_y, left_mask, criterion)
        right_gain = information_gain(sorted_y, right_mask, criterion)
        
        weighted_gain = (left_gain * len(sorted_y[left_mask]) + right_gain * len(sorted_y[right_mask])) / len(sorted_y)
        
        if weighted_gain > best_gain:
            best_gain = weighted_gain
            best_split = split_value

    return best_split, best_gain

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion, features: pd.Series):
    """
    Function to find the optimal attribute to split about.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    features: pd.Series is a list of all the attributes we have to split upon

    return: attribute to split upon
    """

    # According to wheather the features are real or discrete valued and the criterion, find the attribute from the features series with the maximum information gain (entropy or varinace based on the type of output) or minimum gini index (discrete output).
    best_gain = -float('inf')
    best_feature = None
    best_split_value = None

    for feature in features:
        if X[feature].dtype.kind in 'iufc':  # Check if the feature is continuous
            split_value, gain = best_split_continuous(X[feature], y, criterion)
        else:  # For categorical features
            gain = information_gain(y, X[feature], criterion)
            split_value = None

        if gain > best_gain:
            best_gain = gain
            best_feature = feature
            best_split_value = split_value

    return best_feature, best_split_value


def split_data(X: pd.DataFrame, y: pd.Series, attribute, value)->dict:
    """
    Funtion to split the data according to an attribute.
    If needed you can split this function into 2, one for discrete and one for real valued features.
    You can also change the parameters of this function according to your implementation.

    attribute: attribute/feature to split upon
    value: value of that attribute to split upon

    return: splitted data(Input and output)
    """

    # Split the data based on a particular value of a particular attribute. You may use masking as a tool to split the data.

    if X[attribute].dtype.kind in 'iufc':  # Check if the feature is continuous
        left_mask = X[attribute] <= value
        right_mask = X[attribute] > value
    else:
        left_mask = X[attribute] == value
        right_mask = X[attribute] != value

    X_left, y_left = X[left_mask], y[left_mask]
    X_right, y_right = X[right_mask], y[right_mask]
    return {"X": X_left, "y": y_left}, {"X": X_right, "y": y_right}
