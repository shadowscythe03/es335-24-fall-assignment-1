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
    if criterion == "entropy":
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



#base.py

from dataclasses import dataclass
from typing import Literal
import pandas as pd
import numpy as np
#from tree.utils import *  (not working)

@dataclass
class TreeNode:
    attribute: str = None
    value: float = None
    left = None
    right = None
    output = None

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index", "mse"]
    max_depth: int = 5

    def __init__(self, criterion: Literal["information_gain", "gini_index", "mse"], max_depth=5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.tree = None

    def fit(self, X: pd.DataFrame, y: pd.Series, depth=0) -> TreeNode:
        # Stop splitting if all labels are the same
        if len(y.unique()) == 1:
            return TreeNode(output=y.iloc[0])

        # Stop splitting if we reach the maximum depth
        if depth >= self.max_depth:
            return TreeNode(output=y.mean() if y.dtype.kind in 'fc' else y.mode()[0])

        # Apply one-hot encoding for discrete features
        X_encoded = one_hot_encoding(X)

        if self.criterion == "mse" and check_ifreal(y):
            best_attr, best_value = self._find_best_split_mse(X_encoded, y)
            if best_attr is None:
                return TreeNode(output=y.mean())  # No split found, return mean as output

            X_left, y_left, X_right, y_right = split_data(X_encoded, y, best_attr, best_value)
            left_subtree = self.fit(X_left, y_left, depth + 1)
            right_subtree = self.fit(X_right, y_right, depth + 1)
            return TreeNode(attribute=best_attr, value=best_value, left=left_subtree, right=right_subtree)

        elif self.criterion in ["information_gain", "gini_index"]:
            best_attr = opt_split_attribute(X_encoded, y, self.criterion, X_encoded.columns)
            if best_attr is None:
                return TreeNode(output=y.mode()[0])

            value = X_encoded[best_attr].mean() if check_ifreal(X_encoded[best_attr]) else X_encoded[best_attr].mode()[0]
            X_left, y_left, X_right, y_right = split_data(X_encoded, y, best_attr, value)
            left_subtree = self.fit(X_left, y_left, depth + 1)
            right_subtree = self.fit(X_right, y_right, depth + 1)
            return TreeNode(attribute=best_attr, value=value, left=left_subtree, right=right_subtree)

        else:
            raise ValueError(f"Unsupported criterion: {self.criterion}")

    def _find_best_split_mse(self, X: pd.DataFrame, y: pd.Series):
        best_attr = None
        best_value = None
        best_mse = float('inf')

        for attr in X.columns:
            # For each attribute, find the best split by calculating MSE
            unique_values = np.sort(X[attr].unique())
            for value in unique_values:
                X_left, y_left, X_right, y_right = split_data(X, y, attr, value)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue

                mse_left = np.mean((y_left - y_left.mean()) ** 2)
                mse_right = np.mean((y_right - y_right.mean()) ** 2)
                mse_split = (len(y_left) * mse_left + len(y_right) * mse_right) / len(y)

                if mse_split < best_mse:
                    best_mse = mse_split
                    best_attr = attr
                    best_value = value

        return best_attr, best_value

    def predict_row(self, row: pd.Series, node: TreeNode):
        if node.output is not None:
            return node.output
        if row[node.attribute] <= node.value:
            return self.predict_row(row, node.left)
        else:
            return self.predict_row(row, node.right)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        return X.apply(lambda row: self.predict_row(row, self.tree), axis=1)

    def plot_node(self, node: TreeNode, depth=0) -> None:
        if node.output is not None:
            print(f"{'    ' * depth}Output: {node.output}")
            return
        print(f"{'    ' * depth}If {node.attribute} <= {node.value}:")
        self.plot_node(node.left, depth + 1)
        print(f"{'    ' * depth}Else:")
        self.plot_node(node.right, depth + 1)

    def plot(self) -> None:
        self.plot_node(self.tree)
