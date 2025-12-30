import math
import random
from time import sleep

from prettytable import PrettyTable
import utils
from utils import Colors


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        # Initialize weights and biases with random values between -1 and 1
        self.W1 = [[random.uniform(a=-1, b=1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [[random.uniform(a=-1, b=1) for _ in range(hidden_size)]]
        self.W2 = [[random.uniform(a=-1, b=1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [[random.uniform(a=-1, b=1) for _ in range(output_size)]]

    # --- Mathematical Helper Functions (Manual Matrix Operations) ---

    @staticmethod
    def sigmoid(x):
        # Sigmoid activation function with clipping to prevent overflow
        x = max(min(x, 500), -500)
        return 1.0 / (1.0 + math.exp(-x))

    def apply_sigmoid(self, mat):
        return [[self.sigmoid(v) for v in r] for r in mat]

    @staticmethod
    def mat_add(A, B):
        """
        Adds 2 matrices
        :param A: [[... , ... , ... , ... , ... , ...]]
        :param B: [[... , ... , ... , ... , ... , ...]]
        """
        result = []
        for i in range(len(A)):
            current_row = []
            for j in range(len(A[0])):
                sum_value = A[i][j] + B[i][j]
                current_row.append(sum_value)
            result.append(current_row)

        return result

    @staticmethod
    def mat_sub(A, B):
        """
        Subtracts 2 matrices
        :param A: [[... , ... , ... , ... , ... , ...]]
        :param B: [[... , ... , ... , ... , ... , ...]]
        """
        result = []
        for i in range(len(A)):
            current_row = []
            for j in range(len(A[0])):
                sum_value = A[i][j] - B[i][j]
                current_row.append(sum_value)
            result.append(current_row)

        return result

    @staticmethod
    def mat_mul_element(A, B):
        """
        Easy matrix multiplication!
        :param A: [[... , ...]]
        :param B: [[... , ...]]
        """
        result = []

        for i in range(len(A)):
            current_row = []
            for j in range(len(A[0])):
                mul_value = A[i][j] * B[i][j]

                current_row.append(mul_value)

            result.append(current_row)

        return result

    @staticmethod
    def mat_mul(A, B):
        """
        Hard matrix multiplication!
        """
        result = [[0.0] * len(B[0]) for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    @staticmethod
    def mat_scale(A, s):
        """
        Scale a matrix
        [[X , Y , Z]] -> [[sX , sY , sZ]]
        :param A: [[X , Y , Z]]
        :param s: int
        """
        result = []
        for row in A:
            new_row = []

            for v in row:
                scaled_value = v * s
                new_row.append(scaled_value)

            result.append(new_row)

        return result

    @staticmethod
    def transpose(A):
        rows = len(A)
        cols = len(A[0])
        result = []

        for i in range(cols):
            new_row = []

            for j in range(rows):
                val = A[j][i]
                new_row.append(val)

            result.append(new_row)

        return result

    # --- Core Network Logic ---

    def forward(self, x):
        # Propagate input through hidden layer to output
        self.Z1 = self.mat_add(self.mat_mul(x, self.W1), self.b1)
        self.A1 = self.apply_sigmoid(self.Z1)
        self.Z2 = self.mat_add(self.mat_mul(self.A1, self.W2), self.b2)
        self.A2 = self.apply_sigmoid(self.Z2)
        return self.A2

    def train(self, x, y):
        # 1. Forward Pass
        self.forward(x)

        # 2. Calculate Error
        error = self.mat_sub(y, self.A2)

        # 3. Backpropagation (Calculate Gradients)
        # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
        d_out = self.mat_mul_element(error, [[v * (1 - v) for v in r] for r in self.A2])
        err_h = self.mat_mul(d_out, self.transpose(self.W2))
        d_h = self.mat_mul_element(err_h, [[v * (1 - v) for v in r] for r in self.A1])

        # 4. Update Weights and Biases
        self.W2 = self.mat_add(self.W2, self.mat_scale(self.mat_mul(self.transpose(self.A1), d_out), self.learning_rate))
        self.b2 = self.mat_add(self.b2, self.mat_scale(d_out, self.learning_rate))
        self.W1 = self.mat_add(self.W1, self.mat_scale(self.mat_mul(self.transpose(x), d_h), self.learning_rate))
        self.b1 = self.mat_add(self.b1, self.mat_scale(d_h, self.learning_rate))

        return sum([abs(e) for r in error for e in r])


if __name__ == "__main__":
    # Load and prepare data
    train_X, train_Y, test_X, test_Y, total = utils.load_and_process_data()

    utils.print_section("DATASET INFO (Pure Python)")
    print(f"Total: {total} | Train: {len(train_X)} | Test: {len(test_X)}")

    nn = NeuralNetwork(input_size=2, hidden_size=6, output_size=1)
    epochs = utils.EPOCHS

    utils.print_section(f"STARTING TRAINING ({epochs} Epochs)")

    # Training Loop
    for epoch in range(epochs):
        err = 0
        for i in range(len(train_X)):
            # Train sample by sample (Stochastic Gradient Descent)
            err += nn.train(x=[train_X[i]], y=[train_Y[i]])

        if epoch == 0 or epoch % 1000 == 0 or epoch == epochs - 1:
            avg = err / len(train_X)
            c = Colors.GREEN if avg < 0.1 else (Colors.WARNING if avg < 0.3 else Colors.FAIL)
            print(f"Epoch {str(epoch).zfill(4)} | Error: {c}{avg:.6f}{Colors.ENDC}")

    # Display Final Results
    utils.print_final_weights(nn.W1, nn.W2)

    utils.print_section("TEST RESULTS")
    correct = 0
    t = PrettyTable([f"{Colors.CYAN}Act{Colors.ENDC}", f"{Colors.CYAN}Pred{Colors.ENDC}", "Conf", "Stat"])

    for i in range(len(test_X)):
        prob = nn.forward([test_X[i]])[0][0]
        pred = 1.0 if prob > 0.5 else 0.0
        match = (pred == test_Y[i][0])
        if match: correct += 1
        stat = f"{Colors.GREEN}OK{Colors.ENDC}" if match else f"{Colors.FAIL}NO{Colors.ENDC}"
        t.add_row([test_Y[i][0], pred, f"{prob:.4f}", stat])

    print(t)
    print(f"\n{Colors.BOLD}Accuracy: {correct / len(test_X) * 100:.2f}%{Colors.ENDC}")
