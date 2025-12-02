import os
import sys

import numpy as np

from prettytable import PrettyTable


class Colors:
    HEADER = '\033[95m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate

        self.W1 = np.random.uniform(-1, 1, (input_size, hidden_size))
        self.b1 = np.random.uniform(-1, 1, (1, hidden_size))

        self.W2 = np.random.uniform(-1, 1, (hidden_size, output_size))
        self.b2 = np.random.uniform(-1, 1, (1, output_size))

    @staticmethod
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    def forward(self, input_data):
        self.Z1 = np.dot(input_data, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)

        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)

        return self.A2

    def train(self, inputs, targets):
        self.forward(inputs)
        error = targets - self.A2

        d_output = error * self.sigmoid_derivative(self.A2)
        error_hidden = np.dot(d_output, self.W2.T)
        d_hidden = error_hidden * self.sigmoid_derivative(self.A1)

        self.W2 += self.learning_rate * np.dot(self.A1.T, d_output)
        self.b2 += self.learning_rate * np.sum(d_output, axis=0, keepdims=True)

        self.W1 += self.learning_rate * np.dot(inputs.T, d_hidden)
        self.b1 += self.learning_rate * np.sum(d_hidden, axis=0, keepdims=True)

        return np.mean(np.abs(error))


def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

def print_section(title):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}{Colors.ENDC}")


if __name__ == "__main__":
    filename = resource_path(os.path.join("dataset", "jain.txt"))

    raw_data = []
    with open(filename, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                raw_data.append([float(parts[0]), float(parts[1]), int(float(parts[2]))])

    raw_data = np.array(raw_data)
    np.random.shuffle(raw_data)

    X = raw_data[:, 0:2]
    Y_raw = raw_data[:, 2]

    Y = np.where(Y_raw == 2, 1, 0).reshape(-1, 1)

    X_min = X.min(axis=0)
    X_max = X.max(axis=0)
    X_norm = (X - X_min) / (X_max - X_min)

    split_idx = int(0.7 * len(X_norm))
    X_train, X_test = X_norm[:split_idx], X_norm[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    print_section("DATASET INFO (Powered by NumPy)")
    print(f"Total Samples: {Colors.CYAN}{len(X_norm)}{Colors.ENDC}")
    print(f"Training Set:  {Colors.CYAN}{len(X_train)}{Colors.ENDC}")
    print(f"Testing Set:   {Colors.CYAN}{len(X_test)}{Colors.ENDC}")

    nn = NeuralNetwork(input_size=2, hidden_size=6, output_size=1, learning_rate=0.1)
    epochs = 6000

    print_section(f"STARTING TRAINING ({epochs} Epochs)")

    for epoch in range(epochs):
        err = nn.train(X_train, Y_train)

        if epoch == 0 or epoch % 1000 == 0 or epoch == epochs - 1:
            color = Colors.GREEN if err < 0.1 else (Colors.WARNING if err < 0.3 else Colors.FAIL)
            print(f"Epoch {str(epoch).zfill(4)} | Error: {color}{err:.6f}{Colors.ENDC}")

    print_section("FINAL WEIGHTS")

    t_w1 = PrettyTable()
    t_w1.title = "Hidden Layer Weights (W1)"
    t_w1.field_names = [f"{Colors.CYAN}Input Node{Colors.ENDC}"] + [f"{Colors.CYAN}Hidden {i + 1}{Colors.ENDC}" for
                                                                    i in range(6)]
    for i in range(nn.W1.shape[0]):
        row_vals = [f"{w:.4f}" for w in nn.W1[i]]
        t_w1.add_row([f"Input {i + 1}"] + row_vals)
    print(t_w1)
    print()

    t_w2 = PrettyTable()
    t_w2.title = "Output Layer Weights (W2)"
    t_w2.field_names = [f"{Colors.CYAN}Hidden Node{Colors.ENDC}", f"{Colors.CYAN}Output Weight{Colors.ENDC}"]
    for i in range(nn.W2.shape[0]):
        t_w2.add_row([f"Hidden {i + 1}", f"{nn.W2[i][0]:.4f}"])
    print(t_w2)

    print_section("TEST RESULTS")

    predictions = nn.forward(X_test)

    correct = 0
    t_res = PrettyTable()
    t_res.field_names = [f"{Colors.CYAN}Actual{Colors.ENDC}", f"{Colors.CYAN}Predicted{Colors.ENDC}",
                         f"{Colors.CYAN}Confidence{Colors.ENDC}", f"{Colors.CYAN}Status{Colors.ENDC}"]
    t_res.align = "l"

    for i in range(len(X_test)):
        prob = predictions[i][0]
        pred_class = 1 if prob > 0.5 else 0
        actual_class = int(Y_test[i][0])

        is_match = (pred_class == actual_class)
        if is_match: correct += 1

        status = f"{Colors.GREEN}PASS{Colors.ENDC}" if is_match else f"{Colors.FAIL}FAIL{Colors.ENDC}"
        act_lbl = "Class 2" if actual_class == 1 else "Class 1"
        pred_lbl = "Class 2" if pred_class == 1 else "Class 1"

        t_res.add_row([act_lbl, pred_lbl, f"{prob:.4f}", status])

    print(t_res)

    accuracy = (correct / len(X_test)) * 100
    acc_color = Colors.GREEN if accuracy == 100 else (Colors.WARNING if accuracy > 70 else Colors.FAIL)
    print(f"\n{Colors.BOLD}FINAL ACCURACY: {acc_color}{accuracy:.2f}%{Colors.ENDC}")
