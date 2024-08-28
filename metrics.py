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
    assert y.size>0
    
    tp_tn = (y_hat == y).sum()
    return tp_tn/y.size


def precision(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the precision
    """
    tp = ((y_hat == cls) & (y == cls)).sum()
    tp_fp = (y_hat == cls).sum()
    if tp_fp == 0:  #to prevent zero division
        return 0
    return tp/tp_fp


def recall(y_hat: pd.Series, y: pd.Series, cls: Union[int, str]) -> float:
    """
    Function to calculate the recall
    """
    tp = ((y_hat == cls) & (y == cls)).sum()
    tp_fn = (y == cls).sum()
    if tp_fn == 0:#to prevent zero division
        return 0
    return tp/tp_fn


def rmse(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the root-mean-squared-error(rmse)
    """
    # e = y_hat-y
    # se = e**2
    # mse = se.mean()
    rmse = (((y_hat-y)**2).mean())**0.5
    return rmse


def mae(y_hat: pd.Series, y: pd.Series) -> float:
    """
    Function to calculate the mean-absolute-error(mae)
    """
    mae = ((y_hat-y).abs()).mean()
    return mae