import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

class LogRegression():

    def __init__(self, tree_depth, impurity_metric, X_train, X_test, y_train, y_test, class_names):
        self.tree_depth = tree_depth
        self.impurity_metric = impurity_metric
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_names = class_names

    def trainLR(self):
        self.lr = LogisticRegression(C=100.0, random_state=1, solver='lbfgs')
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

        feature_names = self.X_test.columns

        # Create a pandas Series to view feature weights
        feature_weights = pd.Series(weights, index = feature_names)

        # Print the 10 features with the largest weights in absolute value
        print("Top 10 most influential features:")
        print(feature_weights.abs().sort_values(ascending=False).head(10))

        # Classification Report: A detailed breakdown of performance
        # Precision: Of all positive predictions, how many were actually positive?
        # Recall: Of all actual positives, how many did the model find?
        # F1-score: The harmonic mean of precision and recall.
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=['Not Probe', 'Probe']))

        # Confusion Matrix: A table showing correct vs. incorrect predictions
        # [[True Negative,  False Positive],
        #  [False Negative, True Positive]]
        print("\nConfusion Matrix:")
        print(confusion_matrix(self.y_test, y_pred))


        # Train a model with L1 regularization
        lr_l1 = LogisticRegression(penalty='l1', C=1.0, solver='saga', random_state=1)
        lr_l1.fit(self.X_train, self.y_train)
        print(f"L1 Test Accuracy: {lr_l1.score(self.X_test, self.y_test):.4f}")


        # Train a model with L2 regularization
        lr_l2 = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', random_state=1)
        lr_l2.fit(self.X_train, self.y_train)
        print(f"L2 Test Accuracy: {lr_l2.score(self.X_test, self.y_test):.4f}")

        # Plotting the feature weights
        plt.figure(figsize=(12, 6))
        plt.bar(feature_names, lr_l1.coef_[0], alpha=0.7, label='L1 (Saga)')
        plt.bar(feature_names, lr_l2.coef_[0], alpha=0.7, label='L2 (LBFGS)')
        plt.title('Feature Weights for L1 vs L2 Regularization')
        plt.xticks(rotation=90)
        plt.legend()
        plt.show()

