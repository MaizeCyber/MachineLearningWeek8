from DecisionTree import DecisionTree
from LogisticRegression import LogRegression
from Preprocessing import preprocessDecisionTree
from Preprocessing import preprocessScale
from AdaBoost import AdaBoostModel
from RandomForest import RandomForest
from svm import svm_classify

file = "data/Firewall_Rule_Classification.csv"
classColumn = "Class"

def maketree():
    X_train, X_test, y_train, y_test, class_names = preprocessDecisionTree(file=file, classColumn=classColumn)
    d = DecisionTree(tree_depth=4, impurity_metric="gini", X_train=X_train, X_test=X_test, y_train=y_train,
                     y_test=y_test, class_names=class_names)
    d.build_tree()
    d.show_accuracy()

def makelog():
    X_train, X_test, y_train, y_test, class_names, feature_names = preprocessScale(file=file, classColumn=classColumn)
    print(class_names.tolist())
    L = LogRegression(C = 100, X_train=X_train, X_test=X_test, y_train=y_train,
                     y_test=y_test, class_names=class_names, feature_names=feature_names, max_iter=5000)
    L.trainLR()
    L.predict_and_test()

def make_svm():
    X_train, X_test, y_train, y_test, class_names, feature_names = preprocessScale(file=file, classColumn=classColumn, trainingsize=0.10)

    SVM = svm_classify(C = 1, X_train=X_train, X_test=X_test, y_train=y_train,
                     y_test=y_test, class_names=class_names, feature_names=feature_names)
    SVM.train_svm()
    SVM.test_and_metrics()

def make_ada():
    X_train, X_test, y_train, y_test, class_names = preprocessDecisionTree(file=file, classColumn=classColumn)
    d = AdaBoostModel(n_estimators=300, learning_rate=1.0 ,base_tree_depth=4, impurity_metric="gini", X_train=X_train, X_test=X_test, y_train=y_train,
                     y_test=y_test, class_names=class_names)
    d.build_model()
    d.show_metrics()

def make_forest():
    X_train, X_test, y_train, y_test, class_names = preprocessDecisionTree(file=file, classColumn=classColumn)
    f = RandomForest(n_jobs=2 ,n_estimators=300, learning_rate=1.0, impurity_metric="gini", X_train=X_train, X_test=X_test, y_train=y_train,
                     y_test=y_test, class_names=class_names)
    f.build_model()
    f.show_metrics()


if __name__ == "__main__":
    #makelog()
    #make_svm()
    #maketree()
    make_ada()
    make_forest()

