from matplotlib import pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
import graphviz
import seaborn as sns



# ## Building a decision tree

class DecisionTree():

    def __init__(self, tree_depth, impurity_metric, X_train, X_test, y_train, y_test, class_names):
        self.tree_depth = tree_depth
        self.impurity_metric = impurity_metric
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.class_names = class_names


    def build_tree(self):

        self.tree_model = DecisionTreeClassifier(criterion=self.impurity_metric,
                                    max_depth=self.tree_depth,
                                    random_state=1)
        self.tree_model.fit(self.X_train, self.y_train)

        self.y_pred = self.tree_model.predict(self.X_test)


    def show_accuracy(self):
        # 1. Calculate overall accuracy
        accuracy = accuracy_score(self.y_test, self.y_pred)
        print()
        print(f"METRICS FOR DEPTH {self.tree_depth} AND CRITERION {self.impurity_metric.capitalize()}")

        print(f"Overall Accuracy: {accuracy:.4f}\n")

        # 2. Get a detailed classification report
        # Use the 'class_names' from your pd.factorize() step
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_pred, target_names=self.class_names))

        # 3. Generate and plot a confusion matrix
        print("Confusion Matrix:")
        cm = confusion_matrix(self.y_test, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)

        # Accuracy on the training data
        train_accuracy = self.tree_model.score(self.X_train, self.y_train)

        # Accuracy on the test data (already calculated above)
        test_accuracy = self.tree_model.score(self.X_test, self.y_test)

        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Test Accuracy:     {test_accuracy:.4f}")

        actual_depth = self.tree_model.get_depth()
        print(f"The actual depth of the tree is: {actual_depth}")
        print(f"(The maximum allowed depth was: {self.tree_model.max_depth})")

        if self.tree_depth > 1:
            disp.plot(cmap=plt.cm.Blues)
            plt.title(f"Confusion Matrix for Depth {self.tree_depth} and Criterion {self.impurity_metric}")
            plt.show()

            # 2. Get and plot feature importance
            importances = self.tree_model.feature_importances_
            feature_names = self.X_test.columns

            # Create a DataFrame for better visualization
            feature_importance_df = pd.DataFrame({'feature': feature_names, 'importance': importances})

            # Sort by importance and get the top 10
            top_10_features = feature_importance_df.sort_values(by='importance', ascending=False).head(10)

            # Plot the results
            plt.figure(figsize=(10, 6))
            sns.barplot(x='importance', y='feature', data=top_10_features)
            plt.title(f'Top 10 Most Important Features for Depth {self.tree_depth} and Criterion {self.impurity_metric}')
            plt.xlabel('Importance')
            plt.ylabel('Feature')
            plt.tight_layout()
            plt.show()

            # Assume 'tree_model', 'X', and 'class_names' are from your previous code
            feature_names = self.X_test.columns

            # Generate the graph description in the .dot format
            dot_data = tree.export_graphviz(self.tree_model,
                                            out_file=None, # Set out_file=None
                                            feature_names=feature_names,
                                            class_names=self.class_names,
                                            filled=True,
                                            rounded=True,
                                            special_characters=True)



            # Create a graph from the .dot data
            graph = graphviz.Source(dot_data)

            # Display the graph directly in a Jupyter Notebook
            # graph

            # Or save the graph to a file (e.g., 'decision_tree.png')
            # This automatically creates 'decision_tree.png'
            graph.render(f"decision_tree_{self.tree_depth}_{self.impurity_metric}", format="png", view=False, cleanup=True)

            print(f"Decision tree image saved as decision_tree_{self.tree_depth}_{self.impurity_metric}.png")


