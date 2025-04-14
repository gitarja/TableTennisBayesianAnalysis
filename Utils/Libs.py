import numpy as np
def hanley_mcneil_se(auc_value, y_true):
    n1 = np.sum(y_true == 1)  # Number of positive cases
    n2 = np.sum(y_true == 0)  # Number of negative cases

    # Hanley-McNeil formula for standard error
    Q1 = auc_value / (2 - auc_value)
    Q2 = 2 * auc_value**2 / (1 + auc_value)
    se = np.sqrt((auc_value * (1 - auc_value) + (n1 - 1) * (Q1 - auc_value**2) + (n2 - 1) * (Q2 - auc_value**2)) / (n1 * n2))

    return se