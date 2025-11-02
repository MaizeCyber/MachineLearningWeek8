# coding: utf-8

import time
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import matplotlib
from distutils.version import LooseVersion
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from pydotplus import graph_from_dot_data
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

def test_and_metrics(svm):
    """
    Fits an SVM model, evaluates its performance, and prints relevant metrics
    based on the kernel type.
    """
    print(f"--- Analyzing Model: {svm.__class__.__name__} (kernel='{getattr(svm, 'kernel', 'linear')}') ---")

    print('Fitting svm...')
    start_train_time = time.perf_counter() # Start timer
    svm.fit(X_train_std, y_train_resampled)
    end_train_time = time.perf_counter()   # End timer
    train_time = end_train_time - start_train_time
    print(f"Training completed in {train_time:.4f} seconds.")

    # --- 2. Measure Prediction Time ---
    print('\nPredicting model...')
    start_pred_time = time.perf_counter() # Start timer
    y_pred = svm.predict(X_test_std)
    end_pred_time = time.perf_counter()   # End timer
    pred_time = end_pred_time - start_pred_time
    print(f"Prediction completed in {pred_time:.4f} seconds.")

    # ## 1. Evaluate Performance Metrics (Universal for all kernels)
    print("\n--- Performance ---")
    train_accuracy = svm.score(X_train_std, y_train_resampled)
    test_accuracy = svm.score(X_test_std, y_test)
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")

    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Probe', 'Probe']))

    # Confusion Matrix
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # ## 2. Display Model-Specific Insights
    print("\n--- Model Insights ---")

    # Check if the attribute n_iter_ exists
    if hasattr(svm, 'n_iter_'):
        print(f"Model converged in {svm.n_iter_} iterations.")

    # Use hasattr() to check if the model has coefficients (i.e., is a linear model)
    if hasattr(svm, 'coef_'):
        print("Model is linear. Displaying feature weights:")
        weights = svm.coef_[0]
        feature_names = X.columns
        feature_weights = pd.Series(weights, index=feature_names)

        print("\nTop 10 most influential features:")
        print(feature_weights.abs().sort_values(ascending=False).head(30))
    else:
        # For non-linear kernels (poly, rbf, sigmoid)
        print("Model is non-linear. Feature weights are not directly available.")
        print("Instead, we can look at the number of support vectors, which defines model complexity.")

        if hasattr(svm, 'n_support_'):
            print(f"\nNumber of support vectors for each class: {svm.n_support_}")
            print(f"Total support vectors: {sum(svm.n_support_)}")

    print("-" * 50 + "\n")

# 1. Load your data from the CSV file
df = pd.read_csv('kddcup99_csv.csv')

# 2. Prepare the target variable (y)
# Define the labels that correspond to a "Probe"
probe_labels = ['back', 'land', 'neptune', 'pod', 'smurf', 'teardrop']

# Create the binary target 'y'
# We use .str.contains() with a regex pattern to catch all probe variations
# e.g., 'ipsweep' will match 'ipsweep probe'.
# .astype(int) converts True/False to 1/0.
y = df['label'].str.contains('|'.join(probe_labels), case=False, na=False).astype(int)

# 3. Prepare the feature matrix (X)
# First, drop the original label column to create the initial feature set
features = df.drop('label', axis=1)

# Identify categorical columns that need to be converted to numbers
categorical_cols = ['protocol_type', 'service', 'flag']

# Apply one-hot encoding to convert categorical columns into numerical format
X = pd.get_dummies(features, columns=categorical_cols, drop_first=True)

# Now X and y are ready to be used in a scikit-learn model
# For example, splitting the data into training and testing sets:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Getting subset of data for speed purposes:

X_subsample, _, y_subsample, _ = train_test_split(
    X, y,
    train_size=0.10,  # Or use train_size=100 for a fixed number of samples
    random_state=42,
    stratify=y       # This is the crucial part for maintaining class balance
)

# 4. Verify the shapes of your data
print(f"Shape of feature matrix X subsample: {X_subsample.columns}")
print(f"Shape of target vector y: {y.shape}")
print(f"Shape of target vector y subsample: {y_subsample.shape}")
print("\nFirst 5 rows of X subsample:")
print(X_subsample.head())
print("\nFirst 5 values of y subsample:")
print(y_subsample.head())


X_train, X_test, y_train, y_test = train_test_split(
    X_subsample, y_subsample, test_size=0.2, random_state=1, stratify=y_subsample)
print('Data spit into test and training')

rus = RandomUnderSampler(random_state=42)
X_resampled, y_train_resampled = rus.fit_resample(X_train, y_train)

sc = StandardScaler()

print('Fitting standard scaler...')
sc.fit(X_resampled)

print('transforming model...')
X_train_std = sc.transform(X_resampled)
X_test_std = sc.transform(X_test)
# ## Dealing with the nonlinearly separable case using slack variables

X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train_resampled, y_test))




print("LINEAR KERNEL RESULTS:")
svl = SVC(kernel='linear', C=1.0, random_state=1, cache_size=1000)
test_and_metrics(svl)

print('POLY KERNEL RESULTS:')
svp = SVC(kernel='poly', random_state=1, gamma=0.10, C=10.0, cache_size=1000)
test_and_metrics(svp)

print('SIGMOID KERNEL RESULTS:')
svs = SVC(kernel='sigmoid', random_state=1, gamma=0.10, C=10.0, cache_size=1000)
test_and_metrics(svs)

print('RBF KERNEL RESULTS:')
svc = SVC(kernel='rbf', random_state=1, gamma=0.10, C=10.0, cache_size=1000)
test_and_metrics(svc)












