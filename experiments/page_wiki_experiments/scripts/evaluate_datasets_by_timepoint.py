# stdlib
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

data_dir = Path("data")
workspace = Path("workspace")
workspace.mkdir(parents=True, exist_ok=True)

country = "DE"
VERSION = ""
eval_key = "request_path"

data_temporal = load_from_file(
    data_dir
    / f"uaug_temporal_per_flow_no_cache_no_blacklists_ts_{country}{VERSION}.pkl"
)
max_horizon = min(data_temporal.shape[1], 20)


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


def evaluate_static_models_pretrained(
    train_mod: str,
    temporal_data_train: np.ndarray,
    labels_train: np.ndarray,
    test_mod: str,
    temporal_data_test: np.ndarray,
    labels_test: np.ndarray,
    n_folds: int = 3,
) -> None:
    train_mod = train_mod.replace("/", "_")
    test_mod = test_mod.replace("/", "_")
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
        # "knn",
        # "logistic_regression",
        # "random_forest",
        "xgboost",
    ]:
        bkp_file = (
            workspace
            / f"eval_ts_flow_ho_pr_{len(temporal_data_train) + len(temporal_data_test)}_{arch}_train_{train_mod}_test_{test_mod}.json"
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

        print(" >>> ", arch, train_mod, score["str"]["f1_score_macro"], flush=True)


def cleanup_labels(labels: np.ndarray) -> np.ndarray:
    results = []
    labels = pd.Series(labels)
    counts = labels.value_counts().to_dict()
    for value in labels.values:
        if counts[value] >= 70:
            results.append(value)
        else:
            results.append("other")
    return np.asarray(results)


def balance_data(
    static_data: np.ndarray, temporal_data: np.ndarray, labels: np.ndarray
) -> Tuple:
    new_temporal_data = np.asarray(temporal_data)
    static_cols = static_data.columns
    new_static_data = np.asarray(static_data)
    new_labels = cleanup_labels(np.asarray(labels))

    labels_cnt = pd.Series(new_labels).value_counts()
    good_labels = labels_cnt[labels_cnt <= 1000].index.values

    new_index = []
    for idx, label in enumerate(new_labels):
        if label in good_labels:
            new_index.append(idx)

    return (
        pd.DataFrame(new_static_data[new_index], columns=static_cols),
        new_temporal_data[new_index],
        new_labels[new_index],
    )


def evaluate_by_domain_by_horizon(
    label: str, X: np.ndarray, y: np.ndarray, reverse: bool = False
) -> None:
    min_horizon = 0 if reverse else 1
    for domain in tqdm(common_domains):
        horizon_labels = y.copy()
        horizon_labels[horizon_labels != domain] = "other"

        if horizon_labels.value_counts()[domain] < 5:
            continue

        for horizon in range(min_horizon, max_horizon):
            if reverse:
                horizon_data = X[:, horizon:, :]
            else:
                horizon_data = X[:, :horizon, :]

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
        horizon_train_labels = train_y.copy()
        horizon_test_labels = test_y.copy()

        horizon_train_labels[horizon_train_labels != domain] = "other"
        horizon_test_labels[horizon_test_labels != domain] = "other"

        if horizon_train_labels.value_counts()[domain] < 5:
            continue
        if horizon_test_labels.value_counts()[domain] < 5:
            continue

        for horizon in range(min_horizon, max_horizon):
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


#
data_static = pd.read_csv(
    data_dir
    / f"uaug_temporal_per_flow_no_cache_no_blacklists_static_{country}{VERSION}.csv"
)
data_temporal = load_from_file(
    data_dir
    / f"uaug_temporal_per_flow_no_cache_no_blacklists_ts_{country}{VERSION}.pkl"
)

raw_y = data_static[eval_key]

data_static, data_temporal, y = balance_data(data_static, data_temporal, raw_y)
y = pd.Series(y)

common_domains = list(set(y))
shuffle(common_domains)
print("total labels", y.value_counts())
print(
    data_temporal.shape,
    len(np.unique(y)),
)
print(pd.Series(y).value_counts())


assert len(data_temporal) == len(y)


# Split desktop/mobile
android_ua = data_static[data_static["user_agent"].str.contains("Android")]
ios_ua = data_static[data_static["user_agent"].str.contains("iOS")]
ipad_ua = data_static[data_static["user_agent"].str.contains("iPad")]
iphone_ua = data_static[data_static["user_agent"].str.contains("iPhone")]

mobile_data = pd.concat([android_ua, ios_ua, ipad_ua, iphone_ua])
desktop_data = data_static.drop(mobile_data.index, axis=0)

mobile_data_temporal = data_temporal[mobile_data.index]
desktop_data_temporal = data_temporal[desktop_data.index]

mobile_data_y = mobile_data[eval_key].reset_index(drop=True)
desktop_data_y = desktop_data[eval_key].reset_index(drop=True)

# evaluate full
evaluate_by_domain_by_horizon("full_revho", data_temporal, y, reverse=True)
evaluate_by_domain_by_horizon("full_ho", data_temporal, y, reverse=False)

# evaluate desktop UA
evaluate_by_domain_by_horizon(
    desktop_data_temporal, desktop_data_y, "desk_ho", reverse=False
)
evaluate_by_domain_by_horizon(
    desktop_data_temporal, desktop_data_y, "desk_revho", reverse=True
)

# evaluate mobile UA
evaluate_by_domain_by_horizon(
    mobile_data_temporal, mobile_data_y, "mob_ho", reverse=False
)
evaluate_by_domain_by_horizon(
    mobile_data_temporal, mobile_data_y, "mob_revho", reverse=True
)

# train on desktop, test on mobile
evaluate_by_domain_by_horizon_cross(
    "destomob",
    desktop_data_temporal,
    desktop_data_y,
    mobile_data_temporal,
    mobile_data_y,
    reverse=False,
)
evaluate_by_domain_by_horizon_cross(
    "destomob_rev",
    desktop_data_temporal,
    desktop_data_y,
    mobile_data_temporal,
    mobile_data_y,
    reverse=True,
)

# train on mobile, test on desktop
evaluate_by_domain_by_horizon_cross(
    "mobtodes",
    mobile_data_temporal,
    mobile_data_y,
    desktop_data_temporal,
    desktop_data_y,
    reverse=False,
)
evaluate_by_domain_by_horizon_cross(
    "mobtodes_rev",
    mobile_data_temporal,
    mobile_data_y,
    desktop_data_temporal,
    desktop_data_y,
    reverse=True,
)
