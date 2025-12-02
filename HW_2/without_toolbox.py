import math
import os
import random
import sys

from prettytable import PrettyTable


class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.W1 = self.create_matrix(rows=input_size, cols=hidden_size)
        self.b1 = self.create_matrix(rows=1, cols=hidden_size)
        self.W2 = self.create_matrix(rows=hidden_size, cols=output_size)
        self.b2 = self.create_matrix(rows=1, cols=output_size)

    @staticmethod
    def create_matrix(rows, cols):
        return [[random.uniform(-1, 1) for _ in range(cols)] for _ in range(rows)]

    @staticmethod
    def sigmoid(x):
        if x > 100: return 0.0
        if x < -100: return 1.0
        return 1.0 / (1.0 + math.exp(-x))

    def apply_sigmoid(self, matrix):
        return [[self.sigmoid(val) for val in row] for row in matrix]

    @staticmethod
    def apply_sigmoid_derivative(matrix):
        return [[val * (1.0 - val) for val in row] for row in matrix]

    @staticmethod
    def mat_mul(A, B):
        rows_A, cols_A = len(A), len(A[0])
        rows_B, cols_B = len(B), len(B[0])
        result = [[0.0 for _ in range(cols_B)] for _ in range(rows_A)]
        for i in range(rows_A):
            for j in range(cols_B):
                for k in range(cols_A):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    @staticmethod
    def mat_add(A, B):
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def mat_sub(A, B):
        return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def mat_mul_elementwise(A, B):
        return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def mat_scale(A, scalar):
        return [[val * scalar for val in row] for row in A]

    @staticmethod
    def transpose(A):
        return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

    def forward(self, input_data):
        self.Z1 = self.mat_add(self.mat_mul(input_data, self.W1), self.b1)
        self.A1 = self.apply_sigmoid(self.Z1)
        self.Z2 = self.mat_add(self.mat_mul(self.A1, self.W2), self.b2)
        self.A2 = self.apply_sigmoid(self.Z2)
        return self.A2

    def train(self, inputs, targets):
        self.forward(inputs)
        error = self.mat_sub(targets, self.A2)

        d_output = self.mat_mul_elementwise(error, self.apply_sigmoid_derivative(self.A2))
        error_hidden = self.mat_mul(d_output, self.transpose(self.W2))
        d_hidden = self.mat_mul_elementwise(error_hidden, self.apply_sigmoid_derivative(self.A1))

        self.W2 = self.mat_add(self.W2, self.mat_scale(self.mat_mul(self.transpose(self.A1), d_output), self.learning_rate))
        self.b2 = self.mat_add(self.b2, self.mat_scale(d_output, self.learning_rate))
        self.W1 = self.mat_add(self.W1, self.mat_scale(self.mat_mul(self.transpose(inputs), d_hidden), self.learning_rate))
        self.b1 = self.mat_add(self.b1, self.mat_scale(d_hidden, self.learning_rate))

        return sum([abs(e) for row in error for e in row]) / len(error)

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

def print_section(title):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}{Colors.ENDC}")

if __name__ == "__main__":
    raw_data = []
    filename = resource_path(os.path.join("dataset", "jain.txt"))

    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                raw_data.append([float(parts[0]), float(parts[1]), int(float(parts[2]))])

    random.shuffle(raw_data)
    X, Y = [], []
    for row in raw_data:
        X.append(row[0:2])
        Y.append([1.0 if row[2] == 2 else 0.0])

    col1, col2 = [r[0] for r in X], [r[1] for r in X]
    min1, max1, min2, max2 = min(col1), max(col1), min(col2), max(col2)
    X_norm = [[(r[0] - min1) / (max1 - min1) if max1 != min1 else 0,
               (r[1] - min2) / (max2 - min2) if max2 != min2 else 0] for r in X]

    split_idx = int(0.7 * len(X_norm))
    train_X, train_Y = X_norm[:split_idx], Y[:split_idx]
    test_X, test_Y = X_norm[split_idx:], Y[split_idx:]

    print_section("DATASET INFO (Powered by Pure python)")
    print(f"Total Samples: {Colors.CYAN}{len(X_norm)}{Colors.ENDC}")
    print(f"Training Set:  {Colors.CYAN}{len(train_X)}{Colors.ENDC}")
    print(f"Testing Set:   {Colors.CYAN}{len(test_X)}{Colors.ENDC}")

    nn = NeuralNetwork(input_size=2, hidden_size=6, output_size=1, learning_rate=0.1)
    epochs = 6000

    print_section(f"STARTING TRAINING ({epochs} Epochs)")

    for epoch in range(epochs):
        epoch_error = 0
        for i in range(len(train_X)):
            epoch_error += nn.train([train_X[i]], [train_Y[i]])

        if epoch == 0 or epoch % 500 == 0 or epoch == epochs - 1:
            avg_err = epoch_error / len(train_X)
            color = Colors.GREEN if avg_err < 0.1 else (Colors.WARNING if avg_err < 0.3 else Colors.FAIL)
            print(f"Epoch {str(epoch).zfill(4)} | Error: {color}{avg_err:.6f}{Colors.ENDC}")

    print_section("FINAL WEIGHTS")

    t_w1 = PrettyTable()
    t_w1.title = "Hidden Layer Weights (W1)"
    t_w1.field_names = [f"{Colors.CYAN}Input Node{Colors.ENDC}"] + [f"{Colors.CYAN}Hidden {i + 1}{Colors.ENDC}" for i in range(6)]
    for i, row in enumerate(nn.W1):
        t_w1.add_row([f"Input {i + 1}"] + [f"{w:.4f}" for w in row])
    print(t_w1)

    print()
    t_w2 = PrettyTable()
    t_w2.title = "Output Layer Weights (W2)"
    t_w2.field_names = [f"{Colors.CYAN}Hidden Node{Colors.ENDC}", f"{Colors.CYAN}Output Weight{Colors.ENDC}"]
    for i, row in enumerate(nn.W2):
        t_w2.add_row([f"Hidden {i + 1}", f"{row[0]:.4f}"])
    print(t_w2)

    print_section("TEST RESULTS")

    correct = 0
    t_res = PrettyTable()

    t_res.field_names = [
        f"{Colors.CYAN}Actual{Colors.ENDC}",
        f"{Colors.CYAN}Predicted{Colors.ENDC}",
        f"{Colors.CYAN}Confidence{Colors.ENDC}",
        f"{Colors.CYAN}Status{Colors.ENDC}"
    ]
    t_res.align = "l"

    for i in range(len(test_X)):
        prob = nn.forward([test_X[i]])[0][0]
        pred = 1.0 if prob > 0.5 else 0.0
        actual = test_Y[i][0]

        is_match = (pred == actual)
        if is_match: correct += 1

        status = f"{Colors.GREEN}PASS{Colors.ENDC}" if is_match else f"{Colors.FAIL}FAIL{Colors.ENDC}"
        act_lbl = "Class 2" if actual == 1.0 else "Class 1"
        pred_lbl = "Class 2" if pred == 1.0 else "Class 1"

        t_res.add_row([act_lbl, pred_lbl, f"{prob:.4f}", status])

    print(t_res)

    accuracy = (correct / len(test_X)) * 100 if len(test_X) > 0 else 0
    acc_color = Colors.GREEN if accuracy == 100 else (Colors.WARNING if accuracy > 70 else Colors.FAIL)
    print(f"\n{Colors.BOLD}FINAL ACCURACY: {acc_color}{accuracy:.2f}%{Colors.ENDC}")
