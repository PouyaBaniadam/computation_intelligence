import os
import sys
import random

from prettytable import PrettyTable

# Total number of training iterations
# EPOCHS = 1000
# EPOCHS = 2000
EPOCHS = 3000

# ANSI escape codes for terminal text coloring
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

# Resolves file paths for both development and compiled (.exe) environments
def resource_path(relative_path):
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Prints a stylized header for output sections
def print_section(title):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'=' * 60}")
    print(f" {title}")
    print(f"{'=' * 60}{Colors.ENDC}")

# Loads the dataset, processes labels, normalizes features, and splits into train/test
def load_and_process_data(filename="dataset/jain.txt", split_ratio=0.7):
    full_path = resource_path(filename)
    raw_data = []

    # Read data from file
    with open(full_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            raw_data.append([float(parts[0]), float(parts[1]), int(float(parts[2]))])

    # Shuffle data to ensure random distribution
    random.shuffle(raw_data)

    X = []
    Y = []
    for row in raw_data:
        X.append(row[0:2])
        # Map Class 2 -> 1.0, Class 1 -> 0.0
        target = 1.0 if row[2] == 2 else 0.0
        Y.append([target])

    # Manual Min-Max Normalization logic
    col1 = [row[0] for row in X]
    col2 = [row[1] for row in X]
    min1, max1 = min(col1), max(col1)
    min2, max2 = min(col2), max(col2)

    X_norm = []
    for row in X:
        # New_X = (Old_X - X_min) / (X_max - X_min)
        new_row = [
            (row[0] - min1) / (max1 - min1) if max1 != min1 else 0.0,
            (row[1] - min2) / (max2 - min2) if max2 != min2 else 0.0
        ]
        X_norm.append(new_row)

    # Split indices based on the ratio
    split_idx = int(split_ratio * len(X_norm))

    return (X_norm[:split_idx], Y[:split_idx],
            X_norm[split_idx:], Y[split_idx:],
            len(X_norm))

# Displays the trained weights in a formatted table
def print_final_weights(W1, W2):
    print_section("FINAL WEIGHTS")

    # Display W1 (Input -> Hidden)
    t_w1 = PrettyTable()
    t_w1.title = "Hidden Layer Weights (W1)"
    t_w1.field_names = [f"{Colors.CYAN}Input{Colors.ENDC}"] + [f"{Colors.CYAN}H{i + 1}{Colors.ENDC}" for i in range(len(W1[0]))]
    for i, row in enumerate(W1):
        formatted_row = [f"{w:.4f}" for w in row]
        t_w1.add_row([f"In {i + 1}"] + formatted_row)
    print(t_w1)
    print()

    # Display W2 (Hidden -> Output)
    t_w2 = PrettyTable()
    t_w2.title = "Output Layer Weights (W2)"
    t_w2.field_names = [f"{Colors.CYAN}Hidden{Colors.ENDC}", f"{Colors.CYAN}Out{Colors.ENDC}"]
    for i, row in enumerate(W2):
        val = row[0] if isinstance(row, list) else row
        t_w2.add_row([f"H{i + 1}", f"{val:.4f}"])
    print(t_w2)