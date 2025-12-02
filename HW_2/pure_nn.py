import math
import random
from prettytable import PrettyTable
import utils
from utils import Colors


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.W1 = [[random.uniform(a=-1, b=1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.b1 = [[random.uniform(a=-1, b=1) for _ in range(hidden_size)]]
        self.W2 = [[random.uniform(a=-1, b=1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.b2 = [[random.uniform(a=-1, b=1) for _ in range(output_size)]]

    @staticmethod
    def sigmoid(x):
        x = max(min(x, 500), -500)
        return 1.0 / (1.0 + math.exp(-x))

    def apply_sigmoid(self, mat):
        return [[self.sigmoid(v) for v in r] for r in mat]

    @staticmethod
    def mat_mul(A, B):
        result = [[0.0] * len(B[0]) for _ in range(len(A))]
        for i in range(len(A)):
            for j in range(len(B[0])):
                for k in range(len(B)):
                    result[i][j] += A[i][k] * B[k][j]
        return result

    @staticmethod
    def mat_add(A, B):
        return [[A[i][j] + B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def mat_sub(A, B):
        return [[A[i][j] - B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    @staticmethod
    def mat_scale(A, s):
        return [[v * s for v in r] for r in A]

    @staticmethod
    def transpose(A):
        return [[A[j][i] for j in range(len(A))] for i in range(len(A[0]))]

    @staticmethod
    def mat_mul_element(A, B):
        return [[A[i][j] * B[i][j] for j in range(len(A[0]))] for i in range(len(A))]

    def forward(self, x):
        self.Z1 = self.mat_add(self.mat_mul(x, self.W1), self.b1)
        self.A1 = self.apply_sigmoid(self.Z1)
        self.Z2 = self.mat_add(self.mat_mul(self.A1, self.W2), self.b2)
        self.A2 = self.apply_sigmoid(self.Z2)
        return self.A2

    def train(self, x, y):
        self.forward(x)
        error = self.mat_sub(y, self.A2)

        d_out = self.mat_mul_element(error, [[v * (1 - v) for v in r] for r in self.A2])
        err_h = self.mat_mul(d_out, self.transpose(self.W2))
        d_h = self.mat_mul_element(err_h, [[v * (1 - v) for v in r] for r in self.A1])

        self.W2 = self.mat_add(self.W2, self.mat_scale(self.mat_mul(self.transpose(self.A1), d_out), self.learning_rate))
        self.b2 = self.mat_add(self.b2, self.mat_scale(d_out, self.learning_rate))
        self.W1 = self.mat_add(self.W1, self.mat_scale(self.mat_mul(self.transpose(x), d_h), self.learning_rate))
        self.b1 = self.mat_add(self.b1, self.mat_scale(d_h, self.learning_rate))

        return sum([abs(e) for r in error for e in r])


if __name__ == "__main__":
    train_X, train_Y, test_X, test_Y, total = utils.load_and_process_data()

    utils.print_section("DATASET INFO (Pure Python)")
    print(f"Total: {total} | Train: {len(train_X)} | Test: {len(test_X)}")

    nn = NeuralNetwork(input_size=2, hidden_size=6, output_size=1)
    epochs = utils.EPOCHS

    utils.print_section(f"STARTING TRAINING ({epochs} Epochs)")

    for epoch in range(epochs):
        err = 0
        for i in range(len(train_X)):
            err += nn.train(x=[train_X[i]], y=[train_Y[i]])

        if epoch == 0 or epoch % 1000 == 0 or epoch == epochs - 1:
            avg = err / len(train_X)
            c = Colors.GREEN if avg < 0.1 else (Colors.WARNING if avg < 0.3 else Colors.FAIL)
            print(f"Epoch {str(epoch).zfill(4)} | Error: {c}{avg:.6f}{Colors.ENDC}")

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
