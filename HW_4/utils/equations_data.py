import numpy as np

def function_1(x1, x2):
    term1 = 2 * (np.sin(x1) / x1) ** 2
    term2 = 3 * (np.sin(x2) / x2) ** 2
    return np.sqrt(term1 + term2)

def function_2(x1, x2):
    term1 = x1 ** (-2)
    term2 = x2 ** (-1.5)
    return (1 + term1 + term2) ** 2

def generate_dataset(func_id=1, n_samples=800, train_ratio=0.7):
    x1 = np.random.uniform(low=1, high=11, size=n_samples)
    x2 = np.random.uniform(low=1, high=11, size=n_samples)

    match func_id:
        case 1:
            y = function_1(x1, x2)
        case 2:
            y = function_2(x1, x2)
        case _:
            raise NotImplementedError("Only functions `1` | `2` are available!")

    # Combine data in a single matrix
    data = np.column_stack((x1, x2, y))

    n_train = int(n_samples * train_ratio)

    train_data = data[:n_train]
    test_data = data[n_train:]

    return train_data, test_data