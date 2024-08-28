import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

np.random.seed(42)

# Reading the data
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
X = pd.read_csv(url, sep='\s+', header=None,
                 names=["mpg", "cylinders", "displacement", "horsepower", "weight",
                        "acceleration", "model year", "origin", "car name"])

# Clean the above data by removing redundant columns and rows with junk values
# X = pd.read_csv('your_file.csv')
# print(X)
X.replace('?', np.nan, inplace=True)
X.dropna(inplace=True)
X.drop(columns=["car name"], inplace=True)
y = X['mpg']
X.drop(columns=['mpg'],inplace=True)

tree = DecisionTree(criterion='mse')  # Split based on Inf. Gain
tree.fit(X, y)
y_hat = tree.predict(X)
tree.plot()
print("RMSE: ", rmse(y_hat, y))
print("MAE: ", mae(y_hat, y))

# Compare the performance of your model with the decision tree module from scikit learn
# Scikit-learn Decision Tree
sklearn_tree = DecisionTreeRegressor(max_depth=5, random_state=42)
sklearn_tree.fit(X, y)

# Predicting with the scikit-learn Decision Tree
y_pred_sklearn = sklearn_tree.predict(X)

# Calculating accuracy metrics for the scikit-learn Decision Tree
mae = mae(y, y_pred_sklearn)
rmse = rmse(y, y_pred_sklearn)

print(f"Scikit-learn Decision Tree MAE: {mae:.4f}")
print(f"Scikit-learn Decision Tree RMSE: {rmse:.4f}")

# Optionally, plot the scikit-learn Decision Tree
from sklearn.tree import plot_tree

plt.figure(figsize=(20,10))
plot_tree(sklearn_tree, feature_names=X.columns, filled=True)
plt.show()