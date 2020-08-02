#

import numpy as np

# random state initializers

random_seed = 92024
random = np.random.RandomState(random_seed)

train_size = 0.8

# NOTE: calibrated this value to test data
mask_size = 0.2


###########################################################################
# preprocess
###########################################################################

raw_types = {
    "market_id": np.dtype("int32"),
    "order_protocol": np.dtype("int32"),
    "store_primary_category": np.dtype("unicode"),
    "store_id": np.dtype("unicode"),

    "created_at": np.dtype("datetime64"),

    "total_items": np.dtype("float32"),
    "subtotal": np.dtype("float32"),
    "num_distinct_items": np.dtype("float32"),
    "max_item_price": np.dtype("float32"),
    "min_item_price": np.dtype("float32"),

    "total_onshift_dashers": np.dtype("float32"),
    "total_busy_dashers": np.dtype("float32"),
    "total_outstanding_orders": np.dtype("float32"),

    "estimated_order_place_duration": np.dtype("int32"),
    "estimated_store_to_consumer_driving_duration": np.dtype("float32"),

    "actual_delivery_time": np.dtype("datetime64"),
}


###########################################################################
# features, labels, and types
###########################################################################

features = [
    "market_id",  # categorical
    "order_protocol",  # categorical
    "store_primary_category", # lookup
    "store_id", # lookup

    # NOTE: I would like to use DOW but train / test is
    # partitioned by date and train contains an odd selection of
    # weeks
    # "created_at_woy",
    "created_at_dow",
    "created_at_hod",
    "created_at_min",

    "total_items",
    "subtotal",
    "num_distinct_items",
    "max_item_price",
    "min_item_price",

    "total_onshift_dashers",
    "total_busy_dashers",
    "total_outstanding_orders",

    "estimated_order_place_duration",
    "estimated_store_to_consumer_driving_duration",
]
label = "delivery_duration"


allowed_feature_types = [
    np.dtype("float32"), # numeric
    np.dtype("int32"), # categorical
    np.dtype("unicode"), # lookup
]


types = {
    "market_id": np.dtype("int32"),
    "order_protocol": np.dtype("int32"),
    "store_primary_category": np.dtype("unicode"),
    "store_id": np.dtype("unicode"),

    # "created_at_woy": np.dtype("float32"),
    "created_at_dow": np.dtype("float32"),
    "created_at_hod": np.dtype("float32"),
    "created_at_min": np.dtype("float32"),

    "total_items": np.dtype("float32"),
    "subtotal": np.dtype("float32"),
    "num_distinct_items": np.dtype("float32"),
    "max_item_price": np.dtype("float32"),
    "min_item_price": np.dtype("float32"),

    "total_onshift_dashers": np.dtype("float32"),
    "total_busy_dashers": np.dtype("float32"),
    "total_outstanding_orders": np.dtype("float32"),

    # "estimated_order_place_duration": np.dtype("int32"),
    "estimated_order_place_duration": np.dtype("float32"),
    "estimated_store_to_consumer_driving_duration": np.dtype("float32"),

    "delivery_duration": np.dtype("float32"),
}


###########################################################################
# lookups
###########################################################################

lookup_columns = [
    "market_id",
    "order_protocol",
    "created_at_dow",
    "created_at_hod"
]

lookup_values = label
n_pca_components = 5


###########################################################################
# xgboost
###########################################################################

xgboost_early_stopping = 20
xgboost_rounds = 10000

xgboost_params_default = {
    "max_depth":2,
    "eta":1,
    "objective": "reg:squarederror",
    # "booster": "gblinear"
}

xgboost_search_space = {
    "max_depth": [2, 4, 8],
    # "eta": [0.1, 0.2, 0.4],
    "eta": [0.08, 0.2, 0.4],
    "subsample": [0.6, 0.9],
    "colsample_bytree": [0.6, 0.9],
    # "objective": [
    #     "reg:squarederror",
    #     "reg:tweedie",
    # ]
}


###########################################################################
# Random Forest
###########################################################################

random_forest_params = {
    "verbose": 1,
    "n_jobs": -1
}

random_forest_search_space = {
    "max_depth": [10, 20, 50],
    "max_features": [5, 10, 20],
    "min_samples_leaf": [1, 20, 100],
    "max_leaf_nodes": [100, 500, 1000],
}
