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
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler

class svm_classify():

    def __init__(self, C:int, X_train, X_test, y_train, y_test, class_names, feature_names):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_names = class_names
        self.C = C
        self.feature_names = feature_names

    def train_svm(self):

        self.svm = SVC(kernel='linear', C=self.C, random_state=1, cache_size=1000)


    def test_and_metrics(self):
        """
        Fits an SVM model, evaluates its performance, and prints relevant metrics
        based on the kernel type.
        """
        print(f"--- Analyzing Model: {self.svm.__class__.__name__} (kernel='{getattr(self.svm, 'kernel', 'linear')}') ---")

        print('Fitting svm...')
        start_train_time = time.perf_counter() # Start timer
        self.svm.fit(self.X_train, self.y_train)
        end_train_time = time.perf_counter()   # End timer
        train_time = end_train_time - start_train_time
        print(f"Training completed in {train_time:.4f} seconds.")

        # --- 2. Measure Prediction Time ---
        print('\nPredicting model...')
        start_pred_time = time.perf_counter() # Start timer
        y_pred = self.svm.predict(self.X_test)
        end_pred_time = time.perf_counter()   # End timer
        pred_time = end_pred_time - start_pred_time
        print(f"Prediction completed in {pred_time:.4f} seconds.")

        # ## 1. Evaluate Performance Metrics (Universal for all kernels)
        print("\n--- Performance ---")
        train_accuracy = self.svm.score(self.X_train, self.y_train)
        test_accuracy = self.svm.score(self.X_test, self.y_test)
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}")

        # Classification Report

        # Confusion Matrix
        print("Confusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))

        # ## 2. Display Model-Specific Insights
        print("\n--- Model Insights ---")

        # Check if the attribute n_iter_ exists
        if hasattr(self.svm, 'n_iter_'):
            print(f"Model converged in {self.svm.n_iter_} iterations.")

        # Use hasattr() to check if the model has coefficients (i.e., is a linear model)
        if hasattr(self.svm, 'coef_'):
            print("Model is linear. Displaying feature weights:")
            weights = self.svm.coef_[0]

            feature_weights = pd.Series(weights, index=self.feature_names)

            print("\nTop 10 most influential features:")
            print(feature_weights.abs().sort_values(ascending=False).head(10))
        else:
            # For non-linear kernels (poly, rbf, sigmoid)
            print("Model is non-linear. Feature weights are not directly available.")
            print("Instead, we can look at the number of support vectors, which defines model complexity.")

            if hasattr(self.svm, 'n_support_'):
                print(f"\nNumber of support vectors for each class: {self.svm.n_support_}")
                print(f"Total support vectors: {sum(self.svm.n_support_)}")

        print("-" * 50 + "\n")






















