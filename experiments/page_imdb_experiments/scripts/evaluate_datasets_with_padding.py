# stdlib
import argparse
from pathlib import Path
from random import shuffle
import time
from typing import Any
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

base_dir = Path(".")
workspace = base_dir / "workspace_padding"
workspace.mkdir(parents=True, exist_ok=True)

country = "DE"
VERSION = ""
eval_key = "request_path"
data_dir = base_dir / "data"

parser = argparse.ArgumentParser(description="Optional app description")

parser.add_argument("--testcase", type=str, help="Context length")
parser.add_argument("--labels_lim", type=int, help="Labels cnt")
parser.add_argument("--window", type=int, help="Window size")
args = parser.parse_args()

TESTCASE = "entropy"
LABELS_LIM = 1000
WINDOW = 8

data_temporal = load_from_file(
    data_dir
    / f"uaimdbres_{TESTCASE}_temporal_per_flow_no_cache_no_blacklists_ts_{country}_{WINDOW}.pkl"
)
max_horizon = data_temporal.shape[1]
resources_cnt = {
    "api.graphql.imdb.com": 7,
    "m.media-amazon.com": 6,
    "fls-na.amazon.com": 2,
    "images-amazon.com": 3,
    "www.imdb.com": 2,
    "imdb-video.media-imdb.com": 2,
    "m.imdb.com": 2,
    "unagi.amazon.com": 2,
    "dqpnq362acqdi.cloudfront.net": 6,
}
resources_active = set(
    {
        "www.imdb.com",
        "api.graphql.imdb.com",
        "m.media-amazon.com",
        "images-amazon.com",
        "imdb-video.media-imdb.com",
    }
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
            batch_size=1000,
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
        )
    elif arch == "stacked_vae_clf":
        return StackedVAEMLP(
            task_type="classification",
            n_features=temporal_data.shape[-1],
            n_units_embedding=200,
            n_units_out=len(np.unique(labels)),
            output_dropout=0.1,
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
    testname = testname.replace("/", "_")
    input_data = temporal_data.reshape(len(temporal_data), -1)

    labels = encode_labels(raw_labels.copy())
    label_cnt = pd.Series(raw_labels).value_counts()
    assert len(labels[labels == 1]) > 0, label_cnt
    assert len(labels[labels == 0]) > 0, label_cnt
    assert len(labels[labels == 0]) + len(labels[labels == 1]) == len(labels), label_cnt
    assert len(labels[labels == 0]) > len(labels[labels == 1]), label_cnt

    for arch in [
        # "knn",
        # "logistic_regression",
        # "random_forest",
        "xgboost",
    ]:
        bkp_file = (
            workspace
            / f"eval_ts_flow_res_{arch}_{testname}_{TESTCASE}_labels{LABELS_LIM}_window{WINDOW}.json"
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
                time.sleep(1)
                print("static evaluation failed ", arch, e)
                continue
            save_to_file(bkp_file, score)
        print(" >>> ", arch, testname, score["str"]["f1_score_macro"], flush=True)


def cleanup_labels(labels: np.ndarray) -> np.ndarray:
    results = []
    labels = pd.Series(labels)
    counts = labels.value_counts().to_dict()
    for value in labels.values:
        if counts[value] >= 50:
            results.append(value)
        else:
            results.append("other")
    return np.asarray(results)


def balance_data(
    static_data: np.ndarray, temporal_data: np.ndarray, labels: np.ndarray
) -> np.ndarray:
    new_temporal_data = np.asarray(temporal_data)
    static_cols = static_data.columns
    new_static_data = np.asarray(static_data)
    new_labels = cleanup_labels(np.asarray(labels))

    labels_cnt = pd.Series(new_labels).value_counts().head(LABELS_LIM)
    good_labels = labels_cnt[labels_cnt <= 1000].index.values

    new_labels = pd.Series(new_labels)
    new_index = new_labels[new_labels.isin(good_labels)].index.values

    return (
        pd.DataFrame(new_static_data[new_index], columns=static_cols),
        new_temporal_data[new_index],
        new_labels.values[new_index],
    )


def pad_round(x: int, base: int) -> int:
    return int(base * (round(float(x) / base) + 1))


def pad_values(row: int, pad_block: int) -> int:
    if row < 10:
        return row

    return pad_round(row, pad_block)


def evaluate_by_domain_by_horizon(label: str, X: np.ndarray, y: np.ndarray) -> None:
    vfunc = np.vectorize(pad_values)
    for domain in tqdm(common_domains):
        if domain == "other":
            continue
        horizon_labels = pd.Series(y).copy()
        horizon_labels[horizon_labels != domain] = "other"

        if horizon_labels.value_counts()[domain] < 5:
            continue

        # full
        for padblock in [16, 32, 64, 128, 512, 1024]:
            other_lim = 30000
            horizon_data = X.copy()
            horizon_labels = pd.Series(y).copy()
            horizon_labels[horizon_labels != domain] = "other"

            drop_index = (
                horizon_labels[horizon_labels != domain]
                .sample(frac=1, random_state=0)
                .index[other_lim:]
            )
            horizon_labels = horizon_labels.drop(drop_index)
            horizon_data = horizon_data[horizon_labels.index.values]

            horizon_data = vfunc(horizon_data, padblock)

            evaluate_static_models_cv(
                f"{label}_padblock{padblock}_{domain[:20]}",
                horizon_data,
                horizon_labels,
            )


#
data_static = pd.read_csv(
    data_dir
    / f"uaimdbres_{TESTCASE}_temporal_per_flow_no_cache_no_blacklists_static_{country}_{WINDOW}.csv"
)
data_temporal = load_from_file(
    data_dir
    / f"uaimdbres_{TESTCASE}_temporal_per_flow_no_cache_no_blacklists_ts_{country}_{WINDOW}.pkl"
)
max_horizon = data_temporal.shape[1]
raw_y = data_static[eval_key]

data_static, data_temporal, y = balance_data(data_static, data_temporal, raw_y)
y = pd.Series(y)

common_domains = list(set(y))
shuffle(common_domains)

assert len(data_temporal) == len(y)

print(
    data_temporal.shape,
    len(np.unique(y)),
)
print(pd.Series(y).value_counts())


# evaluate full
evaluate_by_domain_by_horizon("fullflow", data_temporal, y)
