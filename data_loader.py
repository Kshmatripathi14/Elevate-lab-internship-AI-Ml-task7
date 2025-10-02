import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def load_data(test_size=0.2, random_state=42, scale=True):
    data = load_breast_cancer()
    X = data.data
    y = data.target
    feature_names = data.feature_names
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    else:
        scaler = None
    return X_train, X_test, y_train, y_test, scaler, feature_names

def reduce_to_2d(X_train, X_test, n_components=2, random_state=42):
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)
    return X_train_2d, X_test_2d, pca
