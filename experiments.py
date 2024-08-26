import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
# ...
def create_fakdata(N,M):
    X = pd.DataFrame({i: pd.Series(np.random.randint(M, size=N), dtype="category") for i in range(5)})
    y = pd.Series(np.random.randint(M, size=N), dtype="category")
    return X, y
# Function to calculate average time (and std) taken by fit() and predict() for different N and M for 4 different cases of DTs
# ...
# Function to plot the results
# ...
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
