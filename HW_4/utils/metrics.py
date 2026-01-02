import numpy as np


def calculate_metrics(y_true, y_pred):
    y_mean = np.mean(y_true)

    # FUV calculation
    numerator_fvu = np.sum((y_pred - y_true) ** 2) # Deduction form: sum of squares of error
    denominator_fvu = np.sum((y_true - y_mean) ** 2) # Denominator: The sum of the data's differences from the mean.
    fvu = numerator_fvu / denominator_fvu

    # PCC calculation
    y_pred_mean = np.mean(y_pred)
    num_pcc = np.sum((y_true - y_mean) * (y_pred - y_pred_mean))
    den_pcc = np.sqrt(np.sum((y_true - y_mean) ** 2) * np.sum((y_pred - y_pred_mean) ** 2))
    pcc = num_pcc / den_pcc

    return fvu, pcc