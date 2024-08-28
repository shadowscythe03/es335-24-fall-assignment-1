import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *
from sklearn.datasets import make_classification, make_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split

np.random.seed(42)
num_average_time = 100  # Number of times to run each experiment to calculate the average values


# Function to create fake data (take inspiration from usage.py)
# ...
def generate_discrete_discrete_data(n_samples, n_features):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=n_features, n_redundant=0, 
                               random_state=42)
    return pd.DataFrame(X), pd.Series(y)

def generate_real_real_data(n_samples, n_features):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                           noise=0.1, random_state=42)
    return pd.DataFrame(X), pd.Series(y)

def generate_real_discrete_data(n_samples, n_features):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=n_features, n_redundant=0, 
                               random_state=42)
    X = pd.DataFrame(X)
    y = pd.Series(y)
    return X, y

def generate_discrete_real_data(n_samples, n_features):
    X, y = make_classification(n_samples=n_samples, n_features=n_features, 
                               n_informative=n_features, n_redundant=0, 
                               random_state=42)
    X = pd.DataFrame(X)
    y = np.random.randn(n_samples)
    return X, pd.Series(y)
# Function to calculate average time (and std) taken by fit() and predict() for different N and M for 4 different cases of DTs
# ...
import matplotlib.pyplot as plt
def measure_runtime(n_samples_list, n_features_list, tree_criterion='mse', n_runs=5):
    learning_times = []
    prediction_times = []

    for n_samples in n_samples_list:
        for n_features in n_features_list:
            print(f"Testing with {n_samples} samples and {n_features} features...")
            
            # Store times for current configuration
            run_learning_times = []
            run_prediction_times = []
            
            for _ in range(n_runs):
                # Generate datasets
                X_discrete, y_discrete = generate_discrete_discrete_data(n_samples, n_features)
                X_real, y_real = generate_real_real_data(n_samples, n_features)
                X_real_discrete, y_real_discrete = generate_real_discrete_data(n_samples, n_features)
                X_discrete_real, y_discrete_real = generate_discrete_real_data(n_samples, n_features)
                
                # Train-test split
                X_train, X_test, y_train, y_test = train_test_split(X_discrete, y_discrete, test_size=0.3, random_state=42)
                
                # Time learning and predicting for Discrete Input, Discrete Output
                start_time = time.time()
                tree = DecisionTree(criterion=tree_criterion, max_depth=5)
                tree.fit(X_train, y_train)
                learning_time = time.time() - start_time
                
                start_time = time.time()
                y_pred = tree.predict(X_test)
                prediction_time = time.time() - start_time
                
                run_learning_times.append(learning_time)
                run_prediction_times.append(prediction_time)

            # Compute mean and standard deviation
            mean_learning_time = np.mean(run_learning_times)
            std_learning_time = np.std(run_learning_times)
            mean_prediction_time = np.mean(run_prediction_times)
            std_prediction_time = np.std(run_prediction_times)
            
            learning_times.append((mean_learning_time, std_learning_time))
            prediction_times.append((mean_prediction_time, std_prediction_time))

            print(f"Learning Time: Mean = {mean_learning_time:.4f} ± {std_learning_time:.4f} seconds")
            print(f"Prediction Time: Mean = {mean_prediction_time:.4f} ± {std_prediction_time:.4f} seconds")
    
    return learning_times, prediction_times

# Define ranges for number of samples and features
n_samples_list = [100, 500, 1000, 5000]
n_features_list = [5, 10, 20, 50]

# Measure runtime
learning_times, prediction_times = measure_runtime(n_samples_list, n_features_list, n_runs=5)

# Convert to arrays for plotting
learning_times_mean = np.array([lt[0] for lt in learning_times])
learning_times_std = np.array([lt[1] for lt in learning_times])
prediction_times_mean = np.array([pt[0] for pt in prediction_times])
prediction_times_std = np.array([pt[1] for pt in prediction_times])


# Plot results


# Function to plot the results
# ...
plt.figure(figsize=(12, 6))

# Learning time plot
plt.subplot(1, 2, 1)
plt.errorbar(n_samples_list, learning_times_mean, yerr=learning_times_std, marker='o', label='Learning Time')
plt.xlabel('Number of Samples')
plt.ylabel('Time (seconds)')
plt.title('Learning Time vs Number of Samples')
plt.legend()

# Prediction time plot
plt.subplot(1, 2, 2)
plt.errorbar(n_samples_list, prediction_times_mean, yerr=prediction_times_std, marker='o', label='Prediction Time')
plt.xlabel('Number of Samples')
plt.ylabel('Time (seconds)')
plt.title('Prediction Time vs Number of Samples')
plt.legend()

plt.tight_layout()
plt.show()
# Other functions
# ...
# Run the functions, Learn the DTs and Show the results/plots
