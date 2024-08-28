# base.py

from dataclasses import dataclass
from typing import Literal, Union

import numpy as np
import pandas as pd
from utils import *
# from graphviz import Digraph

np.random.seed(42)

class TreeNode:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

@dataclass
class DecisionTree:
    criterion: Literal["information_gain", "gini_index", "mse"]
    max_depth: int = 5

    def __init__(self, criterion: str, max_depth: int = 5):
        self.criterion = criterion
        self.max_depth = max_depth
        self.root = None
        self.is_classification = criterion #in ["information_gain", "gini_index"]

    def fit(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Function to train and construct the decision tree.
        """
        self.root = self._build_tree(X, y, depth=0)

    def _build_tree(self, X: pd.DataFrame, y: pd.Series, depth: int) -> TreeNode:
        num_samples, num_features = X.shape

        # Terminate the recursion if the tree is deep enough or there's only one class left
        if depth >= self.max_depth or num_samples <= 1 or len(y.unique()) == 1:
            leaf_value = self._calculate_leaf_value(y)
            return TreeNode(value=leaf_value)

        features = X.columns
        best_feature, best_split_value = opt_split_attribute(X, y, self.criterion, features)

        if best_feature is None:
            leaf_value = self._calculate_leaf_value(y)
            return TreeNode(value=leaf_value)

        left_data, right_data = split_data(X, y, best_feature, best_split_value)

        # If either split is empty, use the parent node's value as the leaf value
        if len(left_data["y"]) == 0 or len(right_data["y"]) == 0:
            leaf_value = self._calculate_leaf_value(y)
            return TreeNode(value=leaf_value)

        left_child = self._build_tree(left_data["X"], left_data["y"], depth + 1)
        right_child = self._build_tree(right_data["X"], right_data["y"], depth + 1)
        return TreeNode(feature=best_feature, threshold=best_split_value, left=left_child, right=right_child)

    def _calculate_leaf_value(self, y: pd.Series) -> Union[float, int]:
        """
        Calculate the leaf value based on the output type.
        For regression (continuous output), return the mean.
        For classification (discrete output), return the mode, or the first element of the mode.
        """
        if y.empty:  # If the series is empty, return None or handle accordingly
            return None
        if self.is_classification:
            mode = y.mode()
            if mode.empty:
                return y.iloc[0]  # Fallback to the first element if mode is empty
            else:
                return mode.iloc[0]
        else:
            return y.mean()

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Function to run the decision tree on test inputs.
        """
        return X.apply(self._predict_single, axis=1)

    def _predict_single(self, x: pd.Series) -> Union[float, int]:
        node = self.root
        while node.left or node.right:
            if node.left is None or node.right is None:
                return node.value
            if x[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def plot(self) -> None:
        """
        Function to plot the tree.
        """
        self._plot_tree(self.root, indent="")

    def _plot_tree(self, node: TreeNode, indent: str):
        if node is None:
            return
        if node.value is not None:
            print(f"{indent}Predict: {node.value}")
        else:
            print(f"{indent}?(X[{node.feature}] <= {node.threshold})")
            print(f"{indent}├─ Yes:")
            self._plot_tree(node.left, indent + "│   ")
            print(f"{indent}└─ No:")
            self._plot_tree(node.right, indent + "    ")
