# stdlib
from pathlib import Path
from random import shuffle
from typing import Any, List, Tuple

# third party
import numpy as np
import pandas as pd
from tls_fingerprinting.models.base.nn.mlp import MLP
from tls_fingerprinting.models.base.nn.stacked_vae_mlp import StackedVAEMLP

# models
from tls_fingerprinting.models.base.nn.ts_model import TimeSeriesModel
from tls_fingerprinting.models.base.nn.vae_mlp import VAEMLP
from tls_fingerprinting.models.base.static.knn import KNNClassifier
from tls_fingerprinting.models.base.static.lr import LinearClassifier
from tls_fingerprinting.models.base.static.random_forest import RFClassifier
from tls_fingerprinting.models.base.static.xgb import XGBoostClassifier
from tls_fingerprinting.utils.evaluation import evaluate_classifier
from tls_fingerprinting.utils.serialization import load_from_file, save_to_file
from tqdm import tqdm

base = Path(".")
workspace = base / "workspace"
data_dir = base / "data"

workspace.mkdir(parents=True, exist_ok=True)

country = "DE"
VERSION = ""
eval_key = "domain"


def _get_temporal_arch_mode(
    mode: str, temporal_data: List, observation_times: List, labels: List
) -> TimeSeriesModel:
    return TimeSeriesModel(
        task_type="classification",
        n_static_units_in=0,
        n_temporal_units_in=temporal_data[0].shape[-1],
        n_temporal_window=max([len(tmp) for tmp in observation_times]),
        output_shape=[len(np.unique(labels))],
        mode=mode,
        use_horizon_condition=False,
        n_iter=1000,
        batch_size=10000,
        n_temporal_layers_hidden=2,
        n_temporal_units_hidden=112,
        dropout=0.1,
        lr=2e-4,
    )


def _get_static_arch_mode(
    arch: str, temporal_data: np.ndarray, labels: np.ndarray
) -> Any:
    if arch == "knn":
        return KNNClassifier()
    elif arch == "logistic_regression":
        return LinearClassifier()
    elif arch == "random_forest":
        return RFClassifier()
    elif arch == "xgboost":
        return XGBoostClassifier()
    elif arch == "nn":
        return MLP(
            task_type="classification",
            n_units_in=temporal_data.shape[-1],
            n_units_out=len(np.unique(labels)),
            dropout=0.1,
            batch_size=10000,
            n_iter=500,
            n_iter_print=10,
        )
    elif arch == "vae_clf":
        return VAEMLP(
            task_type="classification",
            n_features=temporal_data.shape[-1],
            n_units_embedding=200,
            n_units_out=len(np.unique(labels)),
            output_dropout=0.1,
            batch_size=10000,
        )
    elif arch == "stacked_vae_clf":
        return StackedVAEMLP(
            task_type="classification",
            n_features=temporal_data.shape[-1],
            n_units_embedding=200,
            n_units_out=len(np.unique(labels)),
            output_dropout=0.1,
            batch_size=10000,
        )
    else:
        raise RuntimeError(arch)


def encode_labels(labels: np.ndarray) -> np.ndarray:
    labels = pd.Series(labels)
    assert (labels == "other").sum() > 0, labels.value_counts()
    labels[labels != "other"] = 1
    labels[labels == "other"] = 0
    return np.asarray(labels).astype(int)


def evaluate_static_models_cv(
    testname: str,
    temporal_data: np.ndarray,
    raw_labels: np.ndarray,
) -> None:
    input_data = temporal_data.copy().reshape(len(temporal_data), -1)
    labels = encode_labels(raw_labels.copy())

    label_cnt = pd.Series(raw_labels).value_counts()
    assert len(labels[labels == 1]) > 0, label_cnt
    assert len(labels[labels == 0]) > 0, label_cnt
    assert len(labels[labels == 0]) + len(labels[labels == 1]) == len(labels), label_cnt
    assert len(labels[labels == 0]) > len(labels[labels == 1]), label_cnt

    for arch in [
        # "nn",
        # "vae_clf",
        # "stacked_vae_clf",
        # "knn",
        # "logistic_regression",
        # "random_forest",
        "xgboost",
    ]:
        bkp_file = (
            workspace / f"eval_ts_flow_ho_{len(temporal_data)}_{arch}_{testname}.json"
        )

        if bkp_file.exists():
            score = load_from_file(bkp_file)
        else:
            print("evaluate", arch, testname)
            model = _get_static_arch_mode(arch, input_data, labels)
            try:
                score = evaluate_classifier(
                    model, X=np.asarray(input_data), Y=np.asarray(labels)
                )
            except BaseException as e:
                print("static evaluation failed ", arch, e)
                continue
            save_to_file(bkp_file, score)
        print(" >>> ", arch, score["str"])


def evaluate_static_models_pretrained(
    train_mod: str,
    temporal_data_train: List,
    labels_train: pd.Series,
    test_mod: str,
    temporal_data_test: List,
    labels_test: pd.Series,
    n_folds: int = 3,
) -> None:
    print(f"Evaluation diff modality train={train_mod} --> test={test_mod}")

    data_train = pd.DataFrame(
        np.copy(temporal_data_train).reshape(len(temporal_data_train), -1)
    )
    labels_train = pd.Series(labels_train).reset_index(drop=True)

    data_test = pd.DataFrame(
        np.copy(temporal_data_test).reshape(len(temporal_data_test), -1)
    )
    labels_test = pd.Series(labels_test).reset_index(drop=True)

    comm_labels = list(set(labels_train) & set(labels_test))

    labels_train[~labels_train.isin(comm_labels)] = "other"
    labels_test[~labels_test.isin(comm_labels)] = "other"

    labels_train = encode_labels(labels_train)
    labels_test = encode_labels(labels_test)

    classes = list(sorted(set(np.ravel(labels_train))))

    for arch in [
        # "nn",
        # "vae_clf",
        # "stacked_vae_clf",
        # "knn",
        # "logistic_regression",
        # "random_forest",
        "xgboost",
    ]:
        bkp_file = (
            workspace
            / f"eval_ts_flow_ho_pr_{len(temporal_data_train) + len(temporal_data_test)}_{arch}_train_{train_mod}_test_{test_mod}.json"  # noqa
        )

        if bkp_file.exists():
            score = load_from_file(bkp_file)
        else:
            model = _get_static_arch_mode(arch, data_train, labels_train).fit(
                data_train, labels_train
            )
            score = evaluate_classifier(
                [model] * n_folds,
                data_test,
                labels_test,
                pretrained=True,
                n_folds=n_folds,
                classes=classes,
            )
            save_to_file(bkp_file, score)

        print()
        print("------------------------------------------------------")
        print()
        print(" >>> ", arch, score["str"], flush=True)


def cleanup_labels(labels: np.ndarray) -> pd.Series:
    results = []
    labels = pd.Series(labels)
    counts = labels.value_counts().to_dict()
    for value in labels.values:
        if counts[value] >= 5:
            results.append(value)
        else:
            results.append("other")
    return pd.Series(results)


def balance_data_cross(
    left_temporal_data: np.ndarray,
    left_labels: np.ndarray,
    right_temporal_data: np.ndarray,
    right_labels: np.ndarray,
) -> Tuple:
    new_left_data = np.asarray(left_temporal_data)
    new_right_data = np.asarray(right_temporal_data)

    new_left_labels = cleanup_labels(np.asarray(left_labels))
    new_right_labels = cleanup_labels(np.asarray(right_labels))

    labels_left_cnt = pd.Series(new_left_labels).value_counts()
    labels_right_cnt = pd.Series(new_right_labels).value_counts()

    good_left_labels = labels_left_cnt[labels_left_cnt <= 20].index.values
    good_right_labels = labels_right_cnt[labels_right_cnt <= 20].index.values

    new_left_index = new_left_labels[
        new_left_labels.isin(good_left_labels)
    ].index.values
    new_right_index = new_right_labels[
        new_right_labels.isin(good_right_labels)
    ].index.values

    return (
        new_left_data[new_left_index],
        new_left_labels[new_left_index],
        new_right_data[new_right_index],
        new_right_labels[new_right_index],
    )


# get common labels
data_with_cache_static = pd.read_csv(
    data_dir
    / f"tranco_temporal_per_flow_cache_no_blacklists_static_{country}{VERSION}.csv"
)
data_with_cache_temporal = load_from_file(
    data_dir / f"tranco_temporal_per_flow_cache_no_blacklists_ts_{country}{VERSION}.pkl"
)
y_with_cache = data_with_cache_static[eval_key]


data_wo_cache_static = pd.read_csv(
    data_dir
    / f"tranco_temporal_per_flow_no_cache_no_blacklists_static_{country}{VERSION}.csv"
)
data_wo_cache_temporal = load_from_file(
    data_dir
    / f"tranco_temporal_per_flow_no_cache_no_blacklists_ts_{country}{VERSION}.pkl"
)
y_wo_cache = data_wo_cache_static[eval_key]

(
    data_wo_cache_temporal,
    y_wo_cache,
    data_with_cache_temporal,
    y_with_cache,
) = balance_data_cross(
    data_wo_cache_temporal,
    y_wo_cache,
    data_with_cache_temporal,
    y_with_cache,
)

data_full_temporal = np.concatenate([data_with_cache_temporal, data_wo_cache_temporal])
y_full = np.concatenate([y_with_cache, y_wo_cache])

print(
    f"""
        cached data shape = {data_with_cache_temporal.shape} cached data labels cnt = {len(np.unique(y_with_cache))}
        nocache data shape = {data_wo_cache_temporal.shape} nocache data labels cnt = {len(np.unique(y_wo_cache))}
        full data shape = {data_full_temporal.shape} full data labels cnt = {len(np.unique(y_full))}
    """
)
common_domains = list(set(y_with_cache) & set(y_wo_cache))
shuffle(common_domains)

max_horizon = min(data_wo_cache_temporal.shape[1], 10)

assert len(data_wo_cache_temporal) == len(y_wo_cache)
assert len(data_full_temporal) == len(y_full)


def evaluate_by_domain_by_horizon(
    label: str,
    data: np.ndarray,
    labels: np.ndarray,
    reverse: bool = False,
) -> None:
    min_horizon = 0 if reverse else 1

    for domain in tqdm(common_domains):
        horizon_labels = pd.Series(labels).copy()
        horizon_labels[horizon_labels != domain] = "other"

        if horizon_labels.value_counts()[domain] < 5:
            continue

        for horizon in range(min_horizon, max_horizon):
            if reverse:
                horizon_data = data[:, horizon:, :]
            else:
                horizon_data = data[:, :horizon, :]
            print("------------------------------------------------------")
            print(
                f" ### {label} Domain = {domain} Horizon = {horizon} data = {horizon_data.shape} labels cnt = {len(np.unique(horizon_labels))}",  # noqa
                flush=True,
            )
            # evaluate as static input
            evaluate_static_models_cv(
                f"{label}_{horizon}_{domain[:15]}",
                horizon_data,
                horizon_labels,
            )


def evaluate_by_domain_by_horizon_cross(
    label: str,
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_X: np.ndarray,
    test_y: np.ndarray,
    reverse: bool = False,
) -> None:
    min_horizon = 0 if reverse else 1
    for domain in tqdm(common_domains):
        horizon_train_labels = pd.Series(train_y).copy()
        horizon_test_labels = pd.Series(test_y).copy()

        horizon_train_labels[horizon_train_labels != domain] = "other"
        horizon_test_labels[horizon_test_labels != domain] = "other"

        if horizon_train_labels.value_counts()[domain] < 5:
            continue
        if horizon_test_labels.value_counts()[domain] < 5:
            continue

        for horizon in range(min_horizon, max_horizon):
            print("------------------------------------------------------")
            print(f" ### {label} Horizon = {horizon} domain = {domain}")
            if reverse:
                horizon_train_data = train_X[:, horizon:, :]
                horizon_test_data = test_X[:, horizon:, :]
            else:
                horizon_train_data = train_X[:, :horizon, :]
                horizon_test_data = test_X[:, :horizon, :]

            evaluate_static_models_pretrained(
                f"{label}_{horizon}_{domain[:15]}",
                horizon_train_data,
                horizon_train_labels,
                "",
                horizon_test_data,
                horizon_test_labels,
            )


# Train mixed -> test mixed
evaluate_by_domain_by_horizon(
    "mixed", data_wo_cache_temporal, y_wo_cache, reverse=False
)
evaluate_by_domain_by_horizon(
    "mixed_rev", data_wo_cache_temporal, y_wo_cache, reverse=True
)

# Train without caching -> test without caching
evaluate_by_domain_by_horizon(
    "nc_ho", data_wo_cache_temporal, y_wo_cache, reverse=False
)
evaluate_by_domain_by_horizon(
    "nc_revho", data_wo_cache_temporal, y_wo_cache, reverse=True
)

# Train with caching -> test with caching
evaluate_by_domain_by_horizon(
    "ca_ho",
    data_with_cache_temporal,
    y_with_cache,
    reverse=False,
)
evaluate_by_domain_by_horizon(
    "ca_revho",
    data_with_cache_temporal,
    y_with_cache,
    reverse=True,
)

# train on no_cache -> test on with_cache
evaluate_by_domain_by_horizon_cross(
    "nctoc_ho",
    data_wo_cache_temporal,
    y_wo_cache,
    data_with_cache_temporal,
    y_with_cache,
    reverse=False,
)
evaluate_by_domain_by_horizon_cross(
    "nctoc_revho",
    data_wo_cache_temporal,
    y_wo_cache,
    data_with_cache_temporal,
    y_with_cache,
    reverse=True,
)

# train on with_cache -> test on no_cache
evaluate_by_domain_by_horizon_cross(
    "ctonc_ho",
    data_with_cache_temporal,
    y_with_cache,
    data_wo_cache_temporal,
    y_wo_cache,
    reverse=False,
)
evaluate_by_domain_by_horizon_cross(
    "ctonc_revho",
    data_with_cache_temporal,
    y_with_cache,
    data_wo_cache_temporal,
    y_wo_cache,
    reverse=True,
)
