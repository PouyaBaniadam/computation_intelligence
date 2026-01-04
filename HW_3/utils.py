import os
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.preprocessing import OneHotEncoder

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def color_acc(value):
    if value >= 95.0:
        return f"{Colors.GREEN}{Colors.BOLD}{value:6.2f}%{Colors.ENDC}"
    elif value >= 85.0:
        return f"{Colors.CYAN}{value:6.2f}%{Colors.ENDC}"
    elif value >= 70.0:
        return f"{Colors.WARNING}{value:6.2f}%{Colors.ENDC}"
    else:
        return f"{Colors.FAIL}{value:6.2f}%{Colors.ENDC}"

class RBFNetwork:
    def __init__(self, n_centers=10, spread=1.0):
        self.n_centers = n_centers
        self.spread = spread
        self.centers = None
        self.weights = None
        self.enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def _rbf_kernel(self, X, center):
        dist = np.linalg.norm(X - center, axis=1)
        return np.exp(-(dist ** 2) / (2 * (self.spread ** 2)))

    def fit(self, X, Y):
        n_clusters = min(self.n_centers, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=None, n_init=10)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_
        G = np.zeros((X.shape[0], n_clusters))
        for i, center in enumerate(self.centers):
            G[:, i] = self._rbf_kernel(X, center)
        y_reshaped = Y.reshape(-1, 1)
        y_onehot = self.enc.fit_transform(y_reshaped)
        identity = np.eye(n_clusters) * 1e-8
        self.weights = np.linalg.inv(G.T @ G + identity) @ G.T @ y_onehot
        self.active_centers = n_clusters

    def predict(self, X):
        G = np.zeros((X.shape[0], self.active_centers))
        for i, center in enumerate(self.centers):
            G[:, i] = self._rbf_kernel(X, center)
        y_pred_onehot = G @ self.weights
        return self.enc.inverse_transform(y_pred_onehot).flatten()

class PNN:
    def __init__(self, spread=0.1):
        self.spread = spread
        self.X_train = None
        self.y_train = None
        self.classes = None

    def fit(self, X, Y):
        self.X_train = X
        self.y_train = Y
        self.classes = np.unique(Y)

    def predict(self, X):
        y_pred = []
        var_const = 2 * (self.spread ** 2)
        for x_sample in X:
            class_scores = []
            for c in self.classes:
                X_c = self.X_train[self.y_train == c]
                diff = X_c - x_sample
                dist_sq = np.sum(diff ** 2, axis=1)
                activations = np.exp(-dist_sq / var_const)
                score = np.sum(activations) / len(X_c)
                class_scores.append(score)
            y_pred.append(self.classes[np.argmax(class_scores)])
        return np.array(y_pred)

def load_dataset_from_file(filename):
    path = os.path.join('dataset', filename)
    if 'iris' in filename.lower():
        iris = load_iris()
        return iris.data, iris.target
    if filename.endswith('.dat'):
        skip_rows = 0
        with open(path, 'r') as f:
            lines = f.readlines()
            for i, line in enumerate(lines):
                if line.strip().lower().startswith('@data'):
                    skip_rows = i + 1
                    break
        df = pd.read_csv(path, skiprows=skip_rows, header=None)
    else:
        df = pd.read_csv(path, sep=r'\s+', header=None, engine='python')
    df.dropna(inplace=True)
    data = df.values
    X = data[:, :-1]
    Y = data[:, -1]
    Y = pd.factorize(Y)[0]
    return X, Y