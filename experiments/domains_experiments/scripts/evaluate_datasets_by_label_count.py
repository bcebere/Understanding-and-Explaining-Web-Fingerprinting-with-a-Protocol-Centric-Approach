# stdlib
import argparse
from pathlib import Path
from random import shuffle
from typing import Any, Tuple
import warnings

# third party
import numpy as np
import pandas as pd
from tls_fingerprinting.models.base.nn.mlp import MLP
from tls_fingerprinting.models.base.nn.stacked_vae_mlp import StackedVAEMLP

# models
from tls_fingerprinting.models.base.nn.vae_mlp import VAEMLP
from tls_fingerprinting.models.base.static.knn import KNNClassifier
from tls_fingerprinting.models.base.static.lr import LinearClassifier
from tls_fingerprinting.models.base.static.random_forest import RFClassifier
from tls_fingerprinting.models.base.static.xgb import XGBoostClassifier
from tls_fingerprinting.utils.evaluation import evaluate_classifier
from tls_fingerprinting.utils.serialization import load_from_file, save_to_file
from tqdm import tqdm

warnings.filterwarnings("ignore")


base = Path(".")
workspace = base / "workspace_labels"
workspace.mkdir(parents=True, exist_ok=True)
data_dir = base / "data"

country = "DE"
VERSION = ""
eval_key = "domain"

parser = argparse.ArgumentParser(description="Optional app description")
parser.add_argument("--labels_lim", type=int, help="Labels limit")
args = parser.parse_args()

LABELS_LIM = args.labels_lim


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
    input_data = temporal_data.copy().reshape(len(temporal_data), -1).astype(float)
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
        bkp_file = workspace / f"eval_labels_{LABELS_LIM}_{arch}_{testname}.json"

        if bkp_file.exists():
            score = load_from_file(bkp_file)
        else:
            model = _get_static_arch_mode(arch, input_data, labels)
            try:
                score = evaluate_classifier(
                    model, X=np.asarray(input_data), Y=np.asarray(labels)
                )
            except BaseException as e:
                print("static evaluation failed ", arch, e)
                continue
            save_to_file(bkp_file, score)
        print(" >>> ", arch, testname, score["str"]["f1_score_macro"])


def cleanup_labels(labels: np.ndarray) -> np.ndarray:
    results = []
    labels = pd.Series(labels)
    counts = labels.value_counts().to_dict()
    for value in labels.values:
        if counts[value] >= 10:
            results.append(value)
        else:
            results.append("other")
    return np.asarray(results)


def balance_data(
    temporal_data: np.ndarray,
    labels: np.ndarray,
) -> Tuple:
    new_data = np.asarray(temporal_data)
    new_labels = cleanup_labels(np.asarray(labels))

    labels_cnt = pd.Series(new_labels).value_counts().head(LABELS_LIM)
    good_labels = labels_cnt[labels_cnt <= 10000].index.values

    new_labels = pd.Series(new_labels)
    new_index = new_labels[new_labels.isin(good_labels)].index.values

    return (
        new_data[new_index],
        new_labels.values[new_index],
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

data_temporal = np.concatenate([data_with_cache_temporal, data_wo_cache_temporal])
y = np.concatenate([y_with_cache, y_wo_cache])

print(pd.Series(y).value_counts())

(data_temporal, y) = balance_data(
    data_temporal,
    y,
)

print(
    data_temporal.shape,
    len(np.unique(y)),
)
common_domains = list(set(y))
shuffle(common_domains)


max_horizon = min(data_wo_cache_temporal.shape[1], 10)

print(
    f"""
    Data len: no cache {len(data_wo_cache_static)} with cache {len(data_with_cache_static)}
    Labels: {len(common_domains)}
    Max horizon: {max_horizon}
"""
)

print(pd.Series(y).value_counts())
assert len(data_wo_cache_temporal) == len(y_wo_cache)


def evaluate_by_domain_by_horizon(
    label: str,
    data: np.ndarray,
    labels: np.ndarray,
) -> None:
    horizon = 5
    data = data[:, :horizon, :]

    for domain in tqdm(common_domains):
        if domain == "other":
            continue

        horizon_labels = pd.Series(labels).copy()
        horizon_labels[horizon_labels != domain] = "other"

        if horizon_labels.value_counts()[domain] < 5:
            continue

        # evaluate as static input
        evaluate_static_models_cv(
            f"{label}_{domain[:15]}",
            data,
            horizon_labels,
        )


# Train without caching -> test without caching
evaluate_by_domain_by_horizon("full", data_temporal, y)
