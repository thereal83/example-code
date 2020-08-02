#

import copy

import pandas as pd
import numpy as np

from example import settings
from example.persist import cache
from example import preprocess
from example import offline
from example import objective


###########################################################################
# Helpers
###########################################################################

def search_params(search_space):
    import itertools
    keys = list(sorted(search_space.keys()))
    combinations = itertools.product(*(search_space[key] for key in sorted(search_space)))
    for params in combinations:
        yield dict(zip(keys, params))


###########################################################################
# Baseline
###########################################################################

def train_baseline():
    frame = preprocess.train()
    remaining = (
        frame[settings.label]
        - np.log(frame["estimated_store_to_consumer_driving_duration"])
    )
    remaining.name = "remaining"
    frame = pd.concat([frame["store_primary_category"], remaining], axis=1)
    model = frame.groupby("store_primary_category")["remaining"].agg(np.median)
    return model


###########################################################################
# XGBoost
###########################################################################

@cache.memoize()
def train_xgboost(
        weighted=True,
        param=settings.xgboost_params_default,
        num_rounds=settings.xgboost_rounds,
        early_stopping_rounds=settings.xgboost_early_stopping,
):
    import xgboost as xgb
    param["eval_metric"] = ["mae", "rmse"]
    features, labels = offline.build("train")
    dtrain = xgb.DMatrix(features, label=labels)
    features, labels = offline.build("dev")
    deval = xgb.DMatrix(features, label=labels)
    evallist = [(dtrain, 'train'), (deval, 'eval')]

    search_space = settings.xgboost_search_space
    best_score, best_model, best_param = 9999, None, param
    for new in search_params(search_space):
        param.update(new)
        print("="*80)
        print(f"param: {param}")
        # TODO (@ebecker): looks like we can set the random state here
        #   if we use the Scikit-Learn Wrapper interface for XGBoost
        bst = xgb.train(
            param,
            dtrain,
            obj=objective.asymmetric_mse_objective if weighted else None,
            num_boost_round=num_rounds,
            evals=evallist,
            feval=objective.asymmetric_rmse_metric if weighted else None,
            early_stopping_rounds=early_stopping_rounds,
        )
        if bst.best_score < best_score:
            best_score = bst.best_score
            best_model = bst
            best_param = copy.deepcopy(param)

    print("="*80)
    print("Overall best XGBoost")
    print(f"param: {best_param}")
    print(f"score: {best_score}")
    return best_model


###########################################################################
# Random Forest
###########################################################################

@cache.memoize()
def train_random_forest(
        params=settings.random_forest_params
):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error

    train_features, train_labels = offline.build("train")
    dev_features, dev_labels = offline.build("dev")

    search_space = settings.random_forest_search_space
    best_score, best_model, best_param = 9999, None, params
    for new in search_params(search_space):
        params.update(new)
        print("="*80)
        print(f"param: {params}")

        # TODO (@ebecker): consider setting random_state
        regr = RandomForestRegressor(**params)
        regr.fit(train_features, train_labels)
        dev_pred, dev_true = regr.predict(dev_features), dev_labels
        score = mean_squared_error(dev_true, dev_pred)
        print(f"score {score}")

        if score < best_score:
            best_score = score
            best_model = regr
            best_param = copy.deepcopy(params)

    print("="*80)
    print("Overall best RandomForest")
    print(f"param: {best_param}")
    print(f"score: {best_score}")
    return best_model


###########################################################################
# PyTorch
###########################################################################

def train_pytorch():
    raise NotImplementedError


###########################################################################
# Main
###########################################################################

def main():
    train_xgboost(weighted=True)
    train_random_forest()
    train_xgboost(weighted=False)
    # train_pytorch()


if __name__ == "__main__":
    main()
