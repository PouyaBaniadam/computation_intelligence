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
        self.n_centers = n_centers # Hidden layer neurons
        self.spread = spread # Gaussian Normal distribution
        self.centers = None
        self.weights = None
        self.enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

    def _rbf_kernel(self, X, center):
        """
        Some kind of activation function for RBF Network
        1 if exactly as the center,
        0 if far away from the center
        """
        dist = np.linalg.norm(X - center, axis=1) # Euclidean distance
        return np.exp(-(dist ** 2) / (2 * (self.spread ** 2))) # Converting Euclidean distance to Gaussian (Normal distribution)

    def fit(self, X, Y):
        # Using KMeans to find the centers list
        n_clusters = min(self.n_centers, len(X))
        kmeans = KMeans(n_clusters=n_clusters, random_state=None, n_init=10)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        # How much each center has distance from every single datapoint
        # Row = Data count || Column = Centers count
        G = np.zeros((X.shape[0], n_clusters))
        for index, center in enumerate(self.centers):
            G[:, index] = self._rbf_kernel(X, center)

        y_reshaped = Y.reshape(-1, 1)
        y_onehot = self.enc.fit_transform(y_reshaped) # Do not accept simple list! So ==> Previous line

        # G * Weights = Y <==> G : ✅ ||| Weights : ❌ ||| Y : ✅
        # We don't have division in matrices. so we need to inverse!
        self.weights = np.linalg.inv(G.T @ G) @ G.T @ y_onehot
        self.active_centers = n_clusters

    def predict(self, X):
        G = np.zeros((X.shape[0], self.active_centers))
        for index, center in enumerate(self.centers):
            G[:, index] = self._rbf_kernel(X, center)
        y_pred_onehot = G @ self.weights
        return self.enc.inverse_transform(y_pred_onehot).flatten()


class PNN:
    def __init__(self, spread=0.1):
        self.spread = spread
        self.X_train = None
        self.Y_train = None
        self.classes = None

    def fit(self, X, Y):
        # PNN is a "lazy learner"; it doesn't learn weights but memorizes the training data.
        self.X_train = X
        self.Y_train = Y
        # Identify all unique class labels (e.g., [0, 1, 2]) to loop through later
        self.classes = np.unique(Y)

    def predict(self, X):
        y_pred = []
        # Pre-calculate the denominator for the Gaussian function (2 * sigma^2) for efficiency
        var_const = 2 * (self.spread ** 2)

        # Iterate through every single sample in the test set
        for x_sample in X:
            class_scores = []

            # Calculate the probability score for each specific class
            for c in self.classes:
                X_c = self.X_train[self.Y_train == c]

                # Calculate squared Euclidean distance between the test sample and All training samples of class 'c'
                diff = X_c - x_sample
                dist_sq = np.sum(diff ** 2, axis=1)

                # Pattern Layer: Apply Gaussian Kernel (Radial Basis Function)
                # Converts distances into similarity scores (closer points = higher value near 1)
                activations = np.exp(-dist_sq / var_const)

                # Summation Layer: Calculate the average similarity (probability density) for this class
                score = np.sum(activations) / len(X_c)
                class_scores.append(score)

            # Decision Layer (Winner-Takes-All): Choose the class with the highest probability score
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
