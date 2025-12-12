import os
import time
import warnings

import numpy as np
import pandas as pd
from prettytable import PrettyTable
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

warnings.filterwarnings("ignore")

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
        # Prevent errors if n_samples < n_centers
        n_clusters = min(self.n_centers, len(X))

        kmeans = KMeans(n_clusters=n_clusters, random_state=None, n_init=10)
        kmeans.fit(X)
        self.centers = kmeans.cluster_centers_

        G = np.zeros((X.shape[0], n_clusters))
        for i, center in enumerate(self.centers):
            G[:, i] = self._rbf_kernel(X, center)

        y_reshaped = Y.reshape(-1, 1)
        y_onehot = self.enc.fit_transform(y_reshaped)

        # Add regularization term (identity * 1e-8) to avoid singular matrix
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

    try:
        # Check the file format based on its extension
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
    except Exception as e:
        print(f"{Colors.FAIL}Failed to read {filename}: {e}{Colors.ENDC}")
        return None, None

files_map = {
    'Flame': 'Flame.txt',
    'Banana': 'banana.dat',
    'Aggregation': 'Aggregation.txt',
    'Iris': 'iris.dat',
    'Wine': 'wine.dat'
}

RUN_COUNT = 30

table = PrettyTable()

table.field_names = [
    f"{Colors.BOLD}{Colors.BLUE}Algorithm{Colors.ENDC}",
    f"{Colors.BOLD}{Colors.BLUE}Dataset{Colors.ENDC}",
    f"{Colors.BOLD}{Colors.BLUE}Parameters{Colors.ENDC}",
    f"{Colors.BOLD}{Colors.BLUE}Avg Acc (%){Colors.ENDC}",
    f"{Colors.BOLD}{Colors.BLUE}Avg Time (s){Colors.ENDC}",
    f"{Colors.BOLD}{Colors.BLUE}Total Time (s){Colors.ENDC}"
]
table.align = "l"

print(f"{Colors.HEADER}Starting execution (Running {RUN_COUNT} times per algorithm)...{Colors.ENDC}\n")

for ds_name, file_name in files_map.items():
    print(f"Processing dataset: {Colors.BOLD}{ds_name}{Colors.ENDC} ...", end="\r")

    X, Y = load_dataset_from_file(file_name)
    if X is None: continue

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # --- MLP ---
    acc_list = []
    time_list = []
    hidden_layer = 10

    for _ in range(RUN_COUNT):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        start = time.time()
        mlp = MLPClassifier(hidden_layer_sizes=(hidden_layer,), max_iter=800, solver='adam')
        mlp.fit(X_train, Y_train)
        pred = mlp.predict(X_test)
        time_list.append(time.time() - start)
        acc_list.append(accuracy_score(Y_test, pred))

    avg_acc = np.mean(acc_list) * 100
    avg_time = np.mean(time_list)
    total_time = np.sum(time_list)

    table.add_row([
        "MLP",
        ds_name,
        f"Hidden={hidden_layer}",
        color_acc(avg_acc),
        f"{avg_time:.4f}",
        f"{total_time:.4f}"
    ])

    # --- RBF ---
    acc_list = []
    time_list = []
    spread_val = 1.0
    if ds_name == 'Banana': spread_val = 2.0
    if ds_name == 'Aggregation': spread_val = 0.5

    neurons = min(30, len(X) // 5)

    for _ in range(RUN_COUNT):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        start = time.time()
        rbf = RBFNetwork(n_centers=neurons, spread=spread_val)
        rbf.fit(X_train, Y_train)
        pred = rbf.predict(X_test)
        time_list.append(time.time() - start)
        acc_list.append(accuracy_score(Y_test, pred))

    avg_acc = np.mean(acc_list) * 100
    avg_time = np.mean(time_list)
    total_time = np.sum(time_list) # Calculate total time

    table.add_row([
        "RBF",
        ds_name,
        f"Spread={spread_val}, Neu={neurons}",
        color_acc(avg_acc),
        f"{avg_time:.4f}",
        f"{total_time:.4f}"
    ])

    # --- PNN ---
    acc_list = []
    time_list = []
    spread_val = 0.5
    if ds_name == 'Iris': spread_val = 0.1

    for _ in range(RUN_COUNT):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
        start = time.time()
        pnn = PNN(spread=spread_val)
        pnn.fit(X_train, Y_train)
        pred = pnn.predict(X_test)
        time_list.append(time.time() - start)
        acc_list.append(accuracy_score(Y_test, pred))

    avg_acc = np.mean(acc_list) * 100
    avg_time = np.mean(time_list)
    total_time = np.sum(time_list)

    table.add_row([
        "PNN",
        ds_name,
        f"Spread={spread_val}",
        color_acc(avg_acc),
        f"{avg_time:.4f}",
        f"{total_time:.4f}"
    ])

    table.add_row(["-" * 5, "-" * 5, "-" * 5, "-" * 5, "-" * 5, "-" * 5])

# Remove the last separator line
table.del_row(len(table._rows) - 1)

print(" " * 60, end="\r")
print(f"\n{Colors.BOLD}Final Results:{Colors.ENDC}")
print(table)
