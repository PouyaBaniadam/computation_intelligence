import matplotlib.pyplot as plt

from HW_4.utils.equations_data import generate_dataset
from HW_4.utils.fuzzy_system import SimpleMamdaniFIS
from HW_4.utils.metrics import calculate_metrics


def run_experiment(func_id):
    train_data, test_data = generate_dataset(func_id=func_id)

    n_partitions = 15

    fis = SimpleMamdaniFIS(n_mf=n_partitions)
    fis.fit(train_data)

    # A: Fuzzy
    y_pred_fuzzy = fis.predict(test_data[:, :2], use_fuzzy_output=True)
    fvu1, pcc1 = calculate_metrics(test_data[:, 2], y_pred_fuzzy)
    print(f"Mode with fuzzy output -> FVU: {fvu1:.4f}, PCC: {pcc1:.4f}")

    # B: Not Fuzzy (Crisp)
    y_pred_crisp = fis.predict(test_data[:, :2], use_fuzzy_output=False)
    fvu2, pcc2 = calculate_metrics(test_data[:, 2], y_pred_crisp)
    print(f"Mode without fuzzy output -> FVU: {fvu2:.4f}, PCC: {pcc2:.4f}")

    plot_results(test_data, y_pred_fuzzy, func_id)

def plot_results(test_data, y_pred, func_id):
    fig = plt.figure(figsize=(12, 5))

    # Real data plot
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(test_data[:, 0], test_data[:, 1], test_data[:, 2], c='blue', marker='.')
    ax1.set_title(f'Real Function {func_id}')
    ax1.set_xlabel('X1')
    ax1.set_ylabel('X2')

    # Fuzzy logic plot
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(test_data[:, 0], test_data[:, 1], y_pred, c='red', marker='.')
    ax2.set_title(f'Fuzzy Model Output {func_id}')
    ax2.set_xlabel('X1')
    ax2.set_ylabel('X2')

    plt.show()


if __name__ == "__main__":
    run_experiment(func_id=1)
    print("\n" + "=" * 30 + "\n")
    run_experiment(func_id=2)
