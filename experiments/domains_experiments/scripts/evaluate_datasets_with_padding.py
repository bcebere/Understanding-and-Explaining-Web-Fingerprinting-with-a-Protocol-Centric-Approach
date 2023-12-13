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
workspace = base / "workspace_padding"
workspace.mkdir(parents=True, exist_ok=True)
data_dir = base / "data"

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
            model = _get_static_arch_mode(arch, input_data, labels)
            try:
                score = evaluate_classifier(
                    model, X=np.asarray(input_data), Y=np.asarray(labels)
                )
            except BaseException as e:
                print("static evaluation failed ", arch, e)
                continue
            save_to_file(bkp_file, score)
        print(" >>> ", arch, testname, score["str"]["f1_score_macro"], flush=True)


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

print(
    data_wo_cache_temporal.shape,
    data_with_cache_temporal.shape,
    len(np.unique(y_wo_cache)),
    len(np.unique(y_with_cache)),
)
common_domains = list(set(y_with_cache) & set(y_wo_cache))
shuffle(common_domains)

max_horizon = min(data_wo_cache_temporal.shape[1], 10)

print(
    f"""
    Data len: no cache {len(data_wo_cache_static)} with cache {len(data_with_cache_static)}
    Labels: {len(common_domains)}
    Max horizon: {max_horizon}
"""
)

assert len(data_wo_cache_temporal) == len(y_wo_cache)


def pad_round(x: float, base: int) -> int:
    return int(base * (round(float(x) / base) + 1))


def pad_values(row: Any, pad_block: int) -> int:
    if row < 10:
        return row

    return pad_round(row, pad_block)


def evaluate_by_domain_by_horizon(
    label: str,
    data: np.ndarray,
    labels: np.ndarray,
    reverse: bool = False,
) -> None:
    for domain in tqdm(common_domains):
        horizon_labels = pd.Series(labels).copy()
        horizon_labels[horizon_labels != domain] = "other"

        if horizon_labels.value_counts()[domain] < 5:
            continue

        for horizon, padlen in [
            (1, 1),  # certificate
            (2, 1),  # client req
            (3, 1),  # server resp
            (1, 2),  # certificate + client req
            (2, 2),  # client req + server resp
            (1, 3),  # cert + req + resp
            (0, max_horizon),  # entire flow
        ]:
            for padblock in [16, 32, 64, 128, 512, 99999]:
                horizon_data = data.copy()
                rho = min(max_horizon, horizon + padlen)
                vfunc = np.vectorize(pad_values)
                horizon_data[:, horizon:rho, :] = vfunc(
                    horizon_data[:, horizon:rho, :], padblock
                )

                # evaluate as static input
                evaluate_static_models_cv(
                    f"{label}_pad{horizon}_{rho}__padblock{padblock}_{domain[:15]}",
                    horizon_data,
                    horizon_labels,
                )


# Train without caching -> test without caching
evaluate_by_domain_by_horizon("full", data_wo_cache_temporal, y_wo_cache, reverse=True)
