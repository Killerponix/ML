import numpy as np
def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred) ** 2)  #2
    print(f"MSE: {mse:.4f}")
    return mse