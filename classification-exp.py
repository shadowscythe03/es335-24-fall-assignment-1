import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification
from sklearn.model_selection import KFold

np.random.seed(42)

# Code given in the question
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

# For plotting
plt.scatter(X[:, 0], X[:, 1], c=y)
# Assuming the DecisionTree and other functions are already defined as per the earlier discussion

# Generate the synthetic dataset
X, y = make_classification(
    n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5
)

# Convert to DataFrame for compatibility with our decision tree implementation
X = pd.DataFrame(X, columns=['feature_1', 'feature_2'])
y = pd.Series(y)

# Split into training and testing sets
X_train, X_test, y_train, y_test = X[:70], X[70:], y[:70], y[70:]

# Create and train the DecisionTree
tree = DecisionTree(criterion="gini_index", max_depth=5)
tree.tree = tree.fit(X_train, y_train)

# Generate a grid of points to plot the decision boundary
x_min, x_max = X['feature_1'].min() - 1, X['feature_1'].max() + 1
y_min, y_max = X['feature_2'].min() - 1, X['feature_2'].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# Predict the class for each point in the grid
grid_points = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=['feature_1', 'feature_2'])
Z = tree.predict(grid_points)
Z = Z.values.reshape(xx.shape)

# Plotting
plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
plt.scatter(X['feature_1'], X['feature_2'], c=y, edgecolor='k', s=20, cmap=plt.cm.RdYlBu)
plt.title("Decision Tree Decision Boundary for depth-5")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

def evaluate_depths(X, y, depths):
    training_accuracies = []
    validation_accuracies = []

    for depth in depths:
        kf = KFold(n_splits=5, shuffle=True, random_state=1)
        fold_train_accuracies = []
        fold_val_accuracies = []

        for train_index, val_index in kf.split(X):
            X_train, X_val = X.iloc[train_index], X.iloc[val_index]
            y_train, y_val = y.iloc[train_index], y.iloc[val_index]
            
            # Train model
            tree = DecisionTree(criterion="gini_index", max_depth=depth)
            tree.tree = tree.fit(X_train, y_train)
            
            # Predictions
            y_train_pred = tree.predict(X_train)
            y_val_pred = tree.predict(X_val)
            
            # Calculate accuracies
            train_acc = accuracy(y_train_pred, y_train)
            val_acc = accuracy(y_val_pred, y_val)
            
            fold_train_accuracies.append(train_acc)
            fold_val_accuracies.append(val_acc)
        
        training_accuracies.append(np.mean(fold_train_accuracies))
        validation_accuracies.append(np.mean(fold_val_accuracies))

    return training_accuracies, validation_accuracies

# Evaluate
depths = range(1, 6)  # Trying depths from 1 to 5
train_acc, val_acc = evaluate_depths(X, y, depths)

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(depths, train_acc, marker='o', label='Training Accuracy', color='blue')
plt.plot(depths, val_acc, marker='o', label='Validation Accuracy', color='red')
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy vs Tree Depth')
plt.legend()
plt.grid(True)
plt.show()

def nested_cross_validation(X, y, outer_depths, inner_depths):
    outer_kf = KFold(n_splits=5, shuffle=True, random_state=1)
    outer_results = []
    best_depths = []

    for outer_train_index, outer_test_index in outer_kf.split(X):
        X_train, X_test = X.iloc[outer_train_index], X.iloc[outer_test_index]
        y_train, y_test = y.iloc[outer_train_index], y.iloc[outer_test_index]
        
        # Inner Cross-Validation to find the best depth
        inner_kf = KFold(n_splits=5, shuffle=True, random_state=1)
        best_depth = None
        best_val_acc = -np.inf

        for depth in inner_depths:
            fold_val_accuracies = []
            
            for inner_train_index, inner_val_index in inner_kf.split(X_train):
                X_inner_train, X_inner_val = X_train.iloc[inner_train_index], X_train.iloc[inner_val_index]
                y_inner_train, y_inner_val = y_train.iloc[inner_train_index], y_train.iloc[inner_val_index]

                # Train model
                tree = DecisionTree(criterion="gini_index", max_depth=depth)
                tree.tree = tree.fit(X_inner_train, y_inner_train)
                
                # Predictions
                y_inner_val_pred = tree.predict(X_inner_val)
                
                # Calculate accuracy
                val_acc = accuracy(y_inner_val_pred, y_inner_val)
                fold_val_accuracies.append(val_acc)
            
            mean_val_acc = np.mean(fold_val_accuracies)
            if mean_val_acc > best_val_acc:
                best_val_acc = mean_val_acc
                best_depth = depth

        # Train the model with the best depth on the outer training set
        tree = DecisionTree(criterion="gini_index", max_depth=best_depth)
        tree.tree = tree.fit(X_train, y_train)

        # Test the model
        y_test_pred = tree.predict(X_test)
        test_acc = accuracy(y_test_pred, y_test)
        outer_results.append(test_acc)
        best_depths.append(best_depth)

    return np.mean(outer_results), np.std(outer_results), best_depths

# Define the range of depths to try
inner_depths = range(1, 6)  # Depths from 1 to 5
outer_depths = inner_depths  # Using the same range for outer cross-validation

# Perform nested cross-validation
mean_test_acc, std_test_acc, optimal_depths = nested_cross_validation(X, y, outer_depths, inner_depths)

print(f"Mean Test Accuracy: {mean_test_acc:.4f}")
print(f"Standard Deviation of Test Accuracy: {std_test_acc:.4f}")
print(f"Optimal Depths for Each Fold: {optimal_depths}")



