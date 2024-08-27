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
# Splitting into train and test sets
X_train, X_test, y_train, y_test = X[:70], X[70:], y[:70], y[70:]

# Creating and training the DecisionTree
tree = DecisionTree(criterion="gini_index", max_depth=5)
tree.tree = tree.fit(X_train, y_train)

# Predicting and evaluating
y_pred = tree.predict(X_test)

# Plotting the decision tree
def plot_decision_tree(node, depth=0, pos=(0, 0.9), width=0.4, ax=None):
    if node.output is not None:
        ax.text(pos[0], pos[1], f"Output: {node.output}", ha='center', va='center',
                bbox=dict(facecolor='white', edgecolor='black'))
        return

    ax.text(pos[0], pos[1], f"{node.attribute} <= {node.value:.2f}", ha='center', va='center',
            bbox=dict(facecolor='lightblue', edgecolor='black'))

    left_pos = (pos[0] - width, pos[1] - 0.1)
    right_pos = (pos[0] + width, pos[1] - 0.1)

    ax.plot([pos[0], left_pos[0]], [pos[1], left_pos[1]], 'k-')
    ax.plot([pos[0], right_pos[0]], [pos[1], right_pos[1]], 'k-')

    plot_decision_tree(node.left, depth + 1, left_pos, width / 2, ax)
    plot_decision_tree(node.right, depth + 1, right_pos, width / 2, ax)

# Plotting the tree
fig, ax = plt.subplots(figsize=(20, 6))
ax.set_axis_off()
plot_decision_tree(tree.tree, ax=ax)
plt.show()
# Calculating accuracy
acc = accuracy(y_pred, y_test)

# Calculating precision and recall for each class
classes = y_test.unique()
precisions = {cls: precision(y_pred, y_test, cls) for cls in classes}
recalls = {cls: recall(y_pred, y_test, cls) for cls in classes}

# Displaying the results
print(f"Accuracy: {acc:.4f}")
for cls in classes:
    print(f"Class {cls}: Precision = {precisions[cls]:.4f}, Recall = {recalls[cls]:.4f}")

# Compare the performance of your model with the decision tree module from scikit learn