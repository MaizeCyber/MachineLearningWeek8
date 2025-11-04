from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder


class RandomForest():

    def __init__(self, n_jobs, n_estimators, learning_rate, impurity_metric,
                 X_train, X_test, y_train, y_test, class_names, random_state=1):

        self.n_jobs = n_jobs
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.impurity_metric = impurity_metric
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_names = class_names
        self.random_state = random_state
        self.forest = None
        self.y_pred = None

    def build_model(self):
        """
        Builds, fits, and predicts using the RandomForest Classifier
        """

        self.forest = RandomForestClassifier(criterion=self.impurity_metric,
                                        n_estimators=self.n_estimators,
                                        random_state=self.random_state,
                                        n_jobs=self.n_jobs)

        # 3. Fit the model and make predictions
        print(f"Building Random Forest model with {self.n_estimators} estimators and {self.n_jobs} jobs...")
        self.forest.fit(self.X_train, self.y_train)
        self.y_pred = self.forest.predict(self.X_test)
        print("Model building complete.")

    def show_metrics(self):
        """
        Calculates and displays accuracy, classification report, confusion matrix,
        and feature importances for the trained Random Forest.
        """
        if self.forest is None or self.y_pred is None:
            print("Model has not been built yet. Please call .build_model() first.")
            return

        # 1. Calculate overall accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print()
        print(f"METRICS FOR RANDOM FOREST MODEL")
        print(f"(Estimators: {self.n_estimators}, LR: {self.learning_rate})")
        print(f"Overall Accuracy: {accuracy:.4f}\n")

        # 2. Get a detailed classification report
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred, target_names=self.class_names))

        # 3. Generate and plot a confusion matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)

        # 4. Accuracy on the training data
        train_accuracy = self.forest.score(self.X_train, self.y_train)
        test_accuracy = self.forest.score(self.X_test, self.y_test)

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy:     {test_accuracy:.4f}\n")

        # 5. Plot Confusion Matrix
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"Random Forst Confusion Matrix ({self.n_estimators} Estimators)")
        plt.savefig('results/RandomForestConfusionMatrix.png', bbox_inches='tight')
        plt.show()

        # 6. Get and plot feature importance

        importances = self.forest.feature_importances_
        feature_names = self.X_test.columns

        # Create a DataFrame for better visualization
        feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

        # Sort by importance and get the top 10
        top_10_features = feature_importance_df.sort_values(by='importance', ascending=False).head(10)

        # Plot the results
        plt.figure(figsize=(10, 6))
        sns.barplot(x='importance', y='feature', data=top_10_features)
        plt.title(f'Top 10 Features for Random Forest ( {self.n_estimators} Estimators)')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.savefig('results/RandomForestFeatures.png', bbox_inches='tight')
        plt.show()
