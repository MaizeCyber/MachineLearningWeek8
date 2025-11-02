from DecisionTree import DecisionTree
from Preprocessing import preprocessDecisionTree


def maketree():
    X_train, X_test, y_train, y_test, class_names = preprocessDecisionTree('Obfuscated-MalMem2022.csv', "Class",
                                                                           "Category")
    d = DecisionTree(tree_depth=3, impurity_metric="gini", X_train=X_train, X_test=X_test, y_train=y_train,
                     y_test=y_test, class_names=class_names)
    d.build_tree()
    d.show_accuracy()



if __name__ == "__main__":
    maketree()