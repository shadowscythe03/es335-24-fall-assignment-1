#metrics.py
from typing import Union
import pandas as pd


def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the accuracy
    """

    """
    The following assert checks if sizes of y_hat and y are equal.
    Students are required to add appropriate assert checks at places to
    ensure that the function does not fail in corner cases.
    """
    assert y_hat.size == y.size
    # TODO: Write here
    pass


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    pass


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    pass


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """

    pass


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    pass

# #Metrics.py

# from typing import Union
# import pandas as pd

# def accuracy(y_hat: pd.Series, y: pd.Series) -> float:
#     assert y_hat.size == y.size
#     assert y_hat.size > 0   #corner cases

#     correct_predictions = (y_hat == y).sum()
#     return correct_predictions / y.size

# def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
#     assert y_hat.size == y.size
#     assert y_hat.size > 0

#     true_positives = ((y_hat == cls) & (y == cls)).sum()
#     predicted_positives = (y_hat == cls).sum()

#     if predicted_positives == 0:
#         return 0.0
#     return true_positives / predicted_positives #TP/(TP+FP)

# def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
#     assert y_hat.size == y.size
#     assert y_hat.size > 0

#     true_positives = ((y_hat == cls) & (y == cls)).sum()
#     actual_positives = (y == cls).sum()

#     if actual_positives == 0:
#         return 0.0
#     return true_positives / actual_positives    #TP/ (TP+FN)

# def rmse(y_hat: pd.Series, y: pd.Series) -> float:
#     assert y_hat.size == y.size
#     assert y_hat.size > 0

#     squared_errors = [(pred - actual) ** 2 for pred, actual in zip(y_hat, y)]
#     mse = sum(squared_errors) / len(squared_errors)
#     rmse = mse ** 0.5
#     return rmse

# def mae(y_hat: pd.Series, y: pd.Series) -> float:
#     assert y_hat.size == y.size
#     assert y_hat.size > 0

#     absolute_errors = [abs(pred - actual) for pred, actual in zip(y_hat, y)]
#     mae = sum(absolute_errors) / len(absolute_errors)
#     return mae