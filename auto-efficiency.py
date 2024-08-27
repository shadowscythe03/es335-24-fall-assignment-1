import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
df = pd.read_csv(url, delim_whitespace=True, header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# df = pd.read_csv('your_file.csv')
# print(df)
df.replace('?', np.nan, inplace=True)
df.dropna(inplace=True)
df.drop(columns=["car name"], inplace=True)
y = df['mpg']
df.drop(columns=['mpg'],inplace=True)
X = df

# Compare the performance of your model with the decision tree module from scikit learn