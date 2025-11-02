import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd


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

def preprocessScale(file, classColumn, *cols_to_exclude):
    df = pd.read_csv(file)
    df[classColumn], class_names = pd.factorize(df[classColumn])

    y = df[classColumn]

    # 3. Prepare the feature matrix (X)
    # First, drop the original label column to create the initial feature set
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

    sc = StandardScaler()
    sc.fit(X_train)
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)

    return X_train_std, X_test_std, y_train, y_test, class_names