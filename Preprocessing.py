import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from matplotlib import pyplot as plt
import matplotlib
import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.impute import SimpleImputer
import graphviz
import seaborn as sns

def preprocessDecisionTree(file, classColumn, *cols_to_exclude):
    df = pd.read_csv(file)

    #df.replace([np.inf, -np.inf], np.nan, inplace=True)

    df[classColumn],class_names = pd.factorize(df[classColumn])

    y = df[classColumn]

    # Identify categorical columns that need to be converted to numbers

    all_columns_to_drop = [classColumn] + list(cols_to_exclude)

    # 2. Drop all specified columns to create the feature set X
    X = df.drop(columns=all_columns_to_drop, axis=1)


    # 4. Verify the shapes of your data
    print(f"Shape of feature matrix X: {X.columns}")
    print(f"Shape of target vector y: {y.shape}")
    print("\nFirst 5 rows of X:")
    print(X.head())
    print("\nFirst 5 values of y:")
    print(y.head())

    # Now X and y are ready to be used in a scikit-learn model
    # For example, splitting the data into training and testing sets:
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print('Class labels:', np.unique(y))

    # Splitting data into 80% training and 20% test data:

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1, stratify=y)

    return X_train, X_test, y_train, y_test, class_names