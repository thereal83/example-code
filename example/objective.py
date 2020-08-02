#

import numpy as np
import xgboost as xgb


###########################################################################
# Objective
###########################################################################

def asymmetric_mse_gradient(predt, dtrain):
    """Compute the gradient for asymmetric MSE."""
    y = dtrain.get_label()

    error = y - predt
    weight = np.ones_like(error)
    is_late = error > 0
    # NOTE: late order is 2x cost of early order
    weight[is_late] = 2.0

    return weight * (predt - y)


def asymmetric_mse_hessian(predt, dtrain):
    """Compute the hessian for asymmetric MSE."""
    y = dtrain.get_label()

    error = y - predt
    weight = np.ones_like(error)
    is_late = error > 0
    # NOTE: late order is 2x cost of early order
    weight[is_late] = 2.0

    return weight


def asymmetric_mse_objective(predt, dtrain):
    """Asymmetric MSE objective."""
    predt[predt < -1] = -1 + 1e-6
    grad = asymmetric_mse_gradient(predt, dtrain)
    hess = asymmetric_mse_hessian(predt, dtrain)
    return grad, hess


###########################################################################
# Metrics
###########################################################################

def asymmetric_rmse_metric(predt, dtrain):
    """Assymetric RMSE metric."""
    y = dtrain.get_label()

    error = y - predt
    weight = np.ones_like(error)
    is_late = error > 0
    # NOTE: late order is 2x cost of early order
    weight[is_late] = 2.0

    amse = weight * np.power(error, 2)
    return "amse", float(np.sqrt(np.sum(amse) / len(y)))


###########################################################################
# Metrics
###########################################################################

def root_mean_squared_error(y_true, y_pred, weight_pos=1.0):
    """Assymetric RMSE metric."""
    error = y_true - y_pred
    weight = np.ones_like(error)
    is_late = error > 0
    # NOTE: late order is 2x cost of early order
    weight[is_late] = weight_pos

    amse = weight * np.power(error, 2)
    return np.sqrt(np.sum(amse) / len(y_true))
