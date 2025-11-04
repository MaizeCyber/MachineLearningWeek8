from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier  # <-- Import AdaBoost


# graphviz is no longer needed as we can't plot an ensemble

class AdaBoostModel():
    """
    A wrapper class for an AdaBoost classifier using Decision Trees as weak learners.
    """

    def __init__(self, base_tree_depth, n_estimators, learning_rate, impurity_metric,
                 X_train, X_test, y_train, y_test, class_names, random_state=1):
        """
        Initialize the AdaBoost model parameters.

        Args:
            base_tree_depth (int): The max_depth for each weak decision tree (stump).
            n_estimators (int): The total number of trees to build in the ensemble.
            learning_rate (float): Shrinks the contribution of each classifier.
            impurity_metric (str): The function to measure the quality of a split (e.g., 'gini', 'entropy').
            X_train, X_test, y_train, y_test: The training and testing data.
            class_names (list): The names of the target classes.
            random_state (int): Seed for reproducibility.
        """
        self.base_tree_depth = base_tree_depth
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.impurity_metric = impurity_metric
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_names = class_names
        self.random_state = random_state
        self.model = None  # Will hold the trained AdaBoost model
        self.y_pred = None  # Will hold the predictions

    def build_model(self):
        """
        Builds, fits, and predicts using the AdaBoost classifier.
        """

        # 1. Define the base learner (the weak decision tree)
        # This is the "stump" that AdaBoost will use repeatedly.
        base_learner = DecisionTreeClassifier(
            criterion=self.impurity_metric,
            max_depth=self.base_tree_depth,
            random_state=self.random_state
        )

        # 2. Define the AdaBoost classifier
        self.model = AdaBoostClassifier(
            estimator=base_learner,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            random_state=self.random_state
        )

        # 3. Fit the model and make predictions
        print(f"Building AdaBoost model with {self.n_estimators} estimators (max_depth={self.base_tree_depth})...")
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        print("Model building complete.")

    def show_metrics(self):
        """
        Calculates and displays accuracy, classification report, confusion matrix,
        and feature importances for the trained AdaBoost model.
        """
        if self.model is None or self.y_pred is None:
            print("Model has not been built yet. Please call .build_model() first.")
            return

        # 1. Calculate overall accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print()
        print(f"METRICS FOR ADABOOST MODEL")
        print(f"(Base Tree Depth: {self.base_tree_depth}, Estimators: {self.n_estimators}, LR: {self.learning_rate})")
        print(f"Overall Accuracy: {accuracy:.4f}\n")

        # 2. Get a detailed classification report
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred, target_names=self.class_names))

        # 3. Generate and plot a confusion matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)

        # 4. Accuracy on the training data
        train_accuracy = self.model.score(self.X_train, self.y_train)
        test_accuracy = self.model.score(self.X_test, self.y_test)

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy:     {test_accuracy:.4f}\n")

        # 5. Plot Confusion Matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"AdaBoost Confusion Matrix (Depth {self.base_tree_depth}, {self.n_estimators} Estimators)")
        plt.savefig('results/AdaBoostConfusionMatrix.png', bbox_inches='tight')
        plt.show()

        # 6. Get and plot feature importance
        # AdaBoost aggregates feature importances from all weak learners
        importances = self.model.feature_importances_
        feature_names = self.X_test.columns

        # Create a DataFrame for better visualization
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

        # Sort by importance and get the top 10
        top_10_features = feature_importance_df.sort_values(by='importance', ascending=False).head(10)

        # Plot the results
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=top_10_features)
        plt.title(f'Top 10 Features for AdaBoost (Depth {self.base_tree_depth}, {self.n_estimators} Estimators)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('results/AdaBoostFeatures.png', bbox_inches='tight')
        plt.show()

        # NOTE: The Graphviz visualization of a single tree is removed,
        # as AdaBoost is an ensemble of many trees. The feature importance
        # plot is the standard way to interpret the model's focus.
        print("Note: AdaBoost is an ensemble model. The feature importance plot")
        print("      represents the aggregated importance across all estimators.")