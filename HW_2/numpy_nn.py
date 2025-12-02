import numpy as np
from prettytable import PrettyTable
import utils
from utils import Colors


class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.lr = learning_rate
        # Initialize weights and biases with random values between -1 and 1
        self.W1 = np.random.uniform(low=-1, high=1, size=(input_size, hidden_size))
        self.b1 = np.random.uniform(low=-1, high=1, size=(1, hidden_size))
        self.W2 = np.random.uniform(low=-1, high=1, size=(hidden_size, output_size))
        self.b2 = np.random.uniform(low=-1, high=1, size=(1, output_size))

    def sigmoid(self, x):
        # Sigmoid activation with clipping to prevent overflow
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def d_sigmoid(self, x):
        # Derivative: f'(x) = f(x) * (1 - f(x))
        return x * (1 - x)

    def forward(self, x):
        # Forward propagation: Input -> Hidden -> Output
        self.Z1 = np.dot(x, self.W1) + self.b1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.sigmoid(self.Z2)
        return self.A2

    def train(self, x, y):
        # 1. Forward Pass
        self.forward(x)

        # 2. Calculate Error
        error = y - self.A2

        # 3. Backpropagation (Vectorized Gradient Calculation)
        d_out = error * self.d_sigmoid(self.A2)
        d_h = np.dot(d_out, self.W2.T) * self.d_sigmoid(self.A1)

        # 4. Update Weights and Biases (Batch Gradient Descent)
        self.W2 += self.lr * np.dot(self.A1.T, d_out)
        self.b2 += self.lr * np.sum(d_out, axis=0, keepdims=True)
        self.W1 += self.lr * np.dot(x.T, d_h)
        self.b1 += self.lr * np.sum(d_h, axis=0, keepdims=True)

        return np.mean(np.abs(error))


if __name__ == "__main__":
    # Load data using utility function
    tx, ty, tex, tey, total = utils.load_and_process_data()

    # Convert lists to NumPy arrays for vectorized operations
    train_X = np.array(tx)
    train_Y = np.array(ty)
    test_X = np.array(tex)
    test_Y = np.array(tey)

    utils.print_section("DATASET INFO (NumPy Version)")
    print(f"Total: {total} | Train: {len(train_X)} | Test: {len(test_X)}")

    nn = NeuralNetwork(input_size=2, hidden_size=6, output_size=1)
    epochs = utils.EPOCHS

    utils.print_section(f"STARTING TRAINING ({epochs} Epochs)")

    for epoch in range(epochs):
        # Batch training: Process entire dataset at once
        err = nn.train(train_X, train_Y)

        if epoch == 0 or epoch % 1000 == 0 or epoch == epochs - 1:
            c = Colors.GREEN if err < 0.1 else (Colors.WARNING if err < 0.3 else Colors.FAIL)
            print(f"Epoch {str(epoch).zfill(4)} | Error: {c}{err:.6f}{Colors.ENDC}")

    # Display Final Weights
    utils.print_final_weights(nn.W1.tolist(), nn.W2.tolist())

    utils.print_section("TEST RESULTS")
    preds = nn.forward(test_X)

    correct = 0
    t = PrettyTable([f"{Colors.CYAN}Act{Colors.ENDC}", f"{Colors.CYAN}Pred{Colors.ENDC}", "Conf", "Stat"])

    for i in range(len(test_X)):
        prob = preds[i][0]
        pred = 1.0 if prob > 0.5 else 0.0
        actual = test_Y[i][0]
        match = (pred == actual)
        if match: correct += 1
        stat = f"{Colors.GREEN}OK{Colors.ENDC}" if match else f"{Colors.FAIL}NO{Colors.ENDC}"
        t.add_row([actual, pred, f"{prob:.4f}", stat])

    print(t)
    print(f"\n{Colors.BOLD}Accuracy: {correct / len(test_X) * 100:.2f}%{Colors.ENDC}")
