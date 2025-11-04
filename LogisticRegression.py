import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class LogRegression():

    def __init__(self, C:int, X_train, X_test, y_train, y_test, class_names, feature_names, max_iter= 1000):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_names = class_names
        self.C = C
        self.feature_names = feature_names
        self.max_iter = max_iter

    def trainLR(self):
        self.lr = LogisticRegression(C=self.C, random_state=1, solver='lbfgs', max_iter = self.max_iter)
        self.lr.fit(self.X_train, self.y_train)

    def predict_and_test(self):
        # 1. Make predictions on the TEST data
        y_pred = self.lr.predict(self.X_test)


        # 2. Evaluate the predictions
        # Accuracy: What percentage of predictions were correct?
        accuracy = accuracy_score(self.y_test, y_pred)
        print(f"Model Accuracy: {accuracy:.4f}")

        train_accuracy = self.lr.score(self.X_train, self.y_train)
        print(f"Training Accuracy: {train_accuracy:.4f}")

        # Get testing accuracy
        test_accuracy = self.lr.score(self.X_test, self.y_test)
        print(f"Testing Accuracy: {test_accuracy:.4f}")


        # Note: For the 'lbfgs' solver, the value is inside a numpy array
        iterations = self.lr.n_iter_[0]
        print(f"Model converged in {iterations} iterations.")

        weights = self.lr.coef_[0]

        feature_names = self.feature_names

        # Create a pandas Series to view feature weights
        feature_weights = pd.Series(weights, index = feature_names)

        # Print the 10 features with the largest weights in absolute value
        print("Top 10 most influential features:")
        print(feature_weights.abs().sort_values(ascending=False).head(10))

        # Classification Report: A detailed breakdown of performance
        # Precision: Of all positive predictions, how many were actually positive?
        # Recall: Of all actual positives, how many did the model find?
        # F1-score: The harmonic mean of precision and recall.

        # Confusion Matrix: A table showing correct vs. incorrect predictions
        # [[True Negative,  False Positive],
        #  [False Negative, True Positive]]
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))



        # Train a model with L2 regularization
        lr_l2 = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=1, max_iter=self.max_iter)
        lr_l2.fit(self.X_train, self.y_train)
        print(f"L2 Test Accuracy: {lr_l2.score(self.X_test, self.y_test):.4f}")


