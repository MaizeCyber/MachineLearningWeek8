from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay


class ComplementNaiveBayes():
    def __init__(self, alpha, force_alpha, fit_prior, class_prior, norm,
                 X_train, X_test, y_train, y_test, class_names):
        self.alpha = alpha
        self.force_alpha = force_alpha
        self.fit_prior = fit_prior
        self.class_prior = class_prior
        self.norm = norm
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_names = class_names

    def build_model(self):
        self.model = ComplementNB(alpha=self.alpha,
                                  force_alpha=self.force_alpha,
                                  fit_prior=self.fit_prior,
                                  class_prior=self.class_prior,
                                  norm=self.norm)
        print(f"Building Naive Bayes model...")
        self.model.fit(self.X_train, self.y_train)
        self.y_pred = self.model.predict(self.X_test)
        print("Model building complete.")

    def show_metrics(self):
        """
        Calculates and displays accuracy, classification report, confusion matrix,
        and feature importances for the trained Naive Bayes.
        """
        if self.model is None or self.y_pred is None:
            print("Model has not been built yet. Please call .build_model() first.")
            return

        # 1. Calculate overall accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print()
        print(f"METRICS FOR COMPLEMENT NAIVE BAYES")
        print(f"Overall Accuracy: {accuracy:.4f}\n")

        # 2. Get a detailed classification report
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred, target_names=self.class_names))

        # 3. Generate and plot a confusion matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(self.y_test, self.y_pred)
        ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)

        # 4. Accuracy on the training data
        train_accuracy = self.model.score(self.X_train, self.y_train)
        test_accuracy = self.model.score(self.X_test, self.y_test)

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy:     {test_accuracy:.4f}\n")

        # 5. Plot Confusion Matrix
        f1 = f1_score(self.y_pred, self.y_test, average="weighted")

        print("Accuracy:", accuracy)
        print("F1 Score:", f1)

