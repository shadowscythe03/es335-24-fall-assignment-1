#Utils.py

import pandas as pd
import numpy as np

def one_hot_encoding(X: pd.DataFrame) -> pd.DataFrame:
    return pd.get_dummies(X)

def check_ifreal(y: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(y)

def entropy(y: pd.Series) -> float:
    probs = y.value_counts(normalize=True)
    return -np.sum(probs * np.log2(probs))

def gini_index(y: pd.Series) -> float:
    probs = y.value_counts(normalize=True)
    return 1 - np.sum(probs ** 2)

def information_gain(X: pd.DataFrame, y: pd.Series, attribute: str, criterion: str) -> float:
    if criterion == "information_gain":
        criterion_func = entropy
    elif criterion == "gini_index":
        criterion_func = gini_index

    total_entropy = criterion_func(y)
    values, counts = np.unique(X[attribute], return_counts=True)
    weighted_entropy = np.sum([(counts[i] / np.sum(counts)) * criterion_func(y[X[attribute] == values[i]]) for i in range(len(values))])

    return total_entropy - weighted_entropy

def opt_split_attribute(X: pd.DataFrame, y: pd.Series, criterion: str, attributes: pd.Index) -> str:
    best_attr = None
    best_gain = -1
    
    for attr in attributes:
        gain = information_gain(X, y, attr, criterion)
        if gain > best_gain:
            best_gain = gain
            best_attr = attr
    
    return best_attr

def split_data(X: pd.DataFrame, y: pd.Series, attribute: str, value: float):
    left_mask = X[attribute] <= value
    right_mask = ~left_mask
    
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]
