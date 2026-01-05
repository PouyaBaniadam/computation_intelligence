import time
import warnings
import numpy as np
from prettytable import PrettyTable
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from utils import Colors, color_acc, RBFNetwork, PNN, load_dataset_from_file

warnings.filterwarnings("ignore")

files_map = {
    'Flame': 'Flame.txt', 'Banana': 'banana.dat',
    'Aggregation': 'Aggregation.txt', 'Iris': 'iris.dat', 'Wine': 'wine.dat'
}
RUN_COUNT = 30

def evaluate_model(model_class, params, X, Y, runs=RUN_COUNT):
    acc_list, time_list = [], []

    for _ in range(runs):
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)

        model = model_class(**params)

        start = time.time()
        model.fit(X_train, Y_train)
        pred = model.predict(X_test)

        if pred.dtype == object and None in pred:
            pred = np.array([p if p is not None else 0 for p in pred]).astype(int)

        time_list.append(time.time() - start)
        acc_list.append(accuracy_score(Y_test, pred))

    return np.mean(acc_list) * 100, np.mean(time_list), np.sum(time_list)


table = PrettyTable()
table.field_names = [f"{Colors.BLUE}Alg{Colors.ENDC}", f"{Colors.BLUE}Dataset{Colors.ENDC}",
                     f"{Colors.BLUE}Params (Neurons/Hidden){Colors.ENDC}",
                     f"{Colors.BLUE}LR / Spread{Colors.ENDC}",
                     f"{Colors.BLUE}Acc (%){Colors.ENDC}",
                     f"{Colors.BLUE}Time(s){Colors.ENDC}"]
table.align = "l"

print(f"{Colors.HEADER}Starting execution ({RUN_COUNT} runs per model)...{Colors.ENDC}\n")

for ds_name, file_name in files_map.items():
    print(f"Dataset: {Colors.BOLD}{ds_name}{Colors.ENDC} ...", end="\r")
    X, Y = load_dataset_from_file(file_name)
    if X is None: continue

    X = StandardScaler().fit_transform(X)

    rbf_spread = 2.0 if ds_name == 'Banana' else (0.5 if ds_name == 'Aggregation' else 1.0)
    pnn_spread = 0.1 if ds_name == 'Iris' else 0.5

    rbf_neurons = min(30, len(X) // 5)
    mlp_hidden = 10
    mlp_lr = 0.1

    models_to_run = [
        ("MLP", MLPClassifier,
         {"hidden_layer_sizes": (mlp_hidden,), "max_iter": 800, "learning_rate_init": mlp_lr},
         f"Hidden={mlp_hidden}", f"LR={mlp_lr}"),

        ("RBF", RBFNetwork,
         {"n_centers": rbf_neurons, "spread": rbf_spread},
         f"Centers={rbf_neurons}", f"Spread={rbf_spread}"),

        ("PNN", PNN,
         {"spread": pnn_spread},
         "All Data", f"Spread={pnn_spread}")
    ]

    for name, clf, params, param_str, lr_spread_str in models_to_run:
        avg_acc, avg_time, total_time = evaluate_model(clf, params, X, Y)
        table.add_row([name, ds_name, param_str, lr_spread_str, color_acc(avg_acc), f"{avg_time:.4f}"])

    table.add_row(["-" * 3, "-" * 5, "-" * 5, "-" * 5, "-" * 5, "-" * 5])

table.del_row(len(table._rows) - 1)
print(f"\n\n{Colors.BOLD}Final Results:{Colors.ENDC}")
print(table)