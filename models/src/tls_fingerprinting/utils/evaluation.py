# stdlib
import copy
from typing import Any, Dict, List, Optional, Tuple, Union

# third party
import numpy as np
import pandas as pd
from pydantic import validate_arguments
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    cohen_kappa_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GroupKFold,
    KFold,
    StratifiedGroupKFold,
    StratifiedKFold,
)
from sklearn.preprocessing import LabelEncoder, label_binarize

# tls_fingerprinting absolute
import tls_fingerprinting.logger as log
from tls_fingerprinting.utils.reproducibility import enable_reproducible_results

clf_supported_metrics = [
    "aucroc_ovo_macro",
    # "aucroc_ovo_weighted",
    # "aucroc_ovr_macro",
    # "aucroc_ovr_micro",
    # "aucroc_ovr_weighted",
    # "aucprc_weighted",
    "aucprc_macro",
    # "aucprc_micro",
    # "accuracy",
    # "f1_score_micro",
    "f1_score_macro",
    # "f1_score_weighted",
    "kappa",
    "kappa_quadratic",
    # "precision_micro",
    "precision_macro",
    # "precision_weighted",
    # "recall_micro",
    "recall_macro",
    # "recall_weighted",
    # "mcc",
    # "fpr_micro",
    # "tpr_micro",
    # "fpr_macro",
    # "tpr_macro",
]
reg_supported_metrics = ["mse", "mae", "r2"]


def evaluate_auc(
    y_test: np.ndarray,
    y_pred_proba: np.ndarray,
    classes: list,
    multi_class: str = "ovr",  # ovo, ovr
    average: str = "micro",  # micro, macro, weighted
) -> Tuple[float, float]:
    """Helper for evaluating AUCROC/AUCPRC for any number of classes."""

    y_test = np.asarray(y_test)
    y_pred_proba = np.asarray(y_pred_proba)
    y_test_proba = label_binarize(y_test, classes=classes)

    nnan = sum(np.ravel(np.isnan(y_pred_proba)))

    if nnan:
        raise ValueError("nan in predictions. aborting")

    n_classes = y_pred_proba.shape[1]

    if n_classes > 2:
        scores = []
        for i in range(n_classes):
            scores.append(
                average_precision_score(
                    y_test_proba[:, i], y_pred_proba[:, i], average=average
                )
            )

        aucprc = np.mean(scores)

        aucroc = roc_auc_score(
            y_test,
            y_pred_proba,
            multi_class=multi_class,
            average=average,
        )
    else:
        aucprc = average_precision_score(y_test, y_pred_proba[:, 1], average=average)

        aucroc = roc_auc_score(
            y_test,
            y_pred_proba[:, 1],
            average=average,
        )

    return aucroc, aucprc


class classifier_metrics:
    """Helper class for evaluating the performance of the classifier.

    Args:
        metric: list, default=["aucroc_ovr_macro",  "aucroc_ovr_micro",  "aucroc_ovo_macro",  "aucroc_ovo_weighted", "aucprc", "accuracy", "f1_score_micro", "f1_score_macro", "f1_score_weighted",  "kappa", "precision_micro", "precision_macro", "precision_weighted", "recall_micro", "recall_macro", "recall_weighted",  "mcc",]
            The type of metric to use for evaluation.
            Potential values:
                - "aucroc_(ovo/ovr)_(micro/macro)" : the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
                - "aucprc" : The average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
                - "accuracy" : Accuracy classification score.
                - "f1_score_micro": F1 score is a harmonic mean of the precision and recall. This version uses the "micro" average: calculate metrics globally by counting the total true positives, false negatives and false positives.
                - "f1_score_macro": F1 score is a harmonic mean of the precision and recall. This version uses the "macro" average: calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
                - "f1_score_weighted": F1 score is a harmonic mean of the precision and recall. This version uses the "weighted" average: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
                - "kappa", "kappa_quadratic":  computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.
                - "precision_micro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(micro) calculates metrics globally by counting the total true positives.
                - "precision_macro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
                - "precision_weighted": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
                - "recall_micro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(micro) calculates metrics globally by counting the total true positives.
                - "recall_macro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
                - "recall_weighted": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
                - "mcc": The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.
    """

    def __init__(self, metric: Union[str, list] = clf_supported_metrics) -> None:
        if isinstance(metric, str):
            self.metrics = [metric]
        else:
            self.metrics = metric

    def get_metric(self) -> Union[str, list]:
        return self.metrics

    def score_proba(
        self, y_test: np.ndarray, y_pred_proba: np.ndarray, classes: list
    ) -> Dict[str, float]:
        if y_test is None or y_pred_proba is None:
            raise RuntimeError("Invalid input for score_proba")

        results = {}
        y_pred = np.argmax(np.asarray(y_pred_proba), axis=1)

        fpr, tpr = self.fpr_tpr(y_test, y_pred_proba, classes)

        for metric in self.metrics:
            if metric == "aucroc_ovo_macro":
                results[metric] = self.roc_auc_score(
                    y_test, y_pred_proba, classes, multi_class="ovo", average="macro"
                )
            elif metric == "aucroc_ovr_weighted":
                results[metric] = self.roc_auc_score(
                    y_test, y_pred_proba, classes, multi_class="ovr", average="weighted"
                )
            elif metric == "aucroc_ovr_macro":
                results[metric] = self.roc_auc_score(
                    y_test, y_pred_proba, classes, multi_class="ovr", average="macro"
                )
            elif metric == "aucroc_ovr_micro":
                results[metric] = self.roc_auc_score(
                    y_test, y_pred_proba, classes, multi_class="ovr", average="micro"
                )
            elif metric == "aucprc_micro":
                results[metric] = self.average_precision_score(
                    y_test, y_pred_proba, classes, average="micro"
                )
            elif metric == "aucprc_macro":
                results[metric] = self.average_precision_score(
                    y_test,
                    y_pred_proba,
                    classes,
                    average="macro",
                )
            elif metric == "aucprc_weighted":
                results[metric] = self.average_precision_score(
                    y_test,
                    y_pred_proba,
                    classes,
                    average="weighted",
                )
            elif metric == "accuracy":
                results[metric] = accuracy_score(y_test, y_pred)
            elif metric == "f1_score_micro":
                results[metric] = f1_score(
                    y_test, y_pred, average="micro", zero_division=0
                )
            elif metric == "f1_score_macro":
                results[metric] = f1_score(
                    y_test, y_pred, average="macro", zero_division=0
                )
            elif metric == "f1_score_weighted":
                results[metric] = f1_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
            elif metric == "kappa":
                results[metric] = cohen_kappa_score(y_test, y_pred)
            elif metric == "kappa_quadratic":
                results[metric] = cohen_kappa_score(y_test, y_pred, weights="quadratic")
            elif metric == "recall_micro":
                results[metric] = recall_score(
                    y_test, y_pred, average="micro", zero_division=0
                )
            elif metric == "recall_macro":
                results[metric] = recall_score(
                    y_test, y_pred, average="macro", zero_division=0
                )
            elif metric == "recall_weighted":
                results[metric] = recall_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
            elif metric == "precision_micro":
                results[metric] = precision_score(
                    y_test, y_pred, average="micro", zero_division=0
                )
            elif metric == "precision_macro":
                results[metric] = precision_score(
                    y_test, y_pred, average="macro", zero_division=0
                )
            elif metric == "precision_weighted":
                results[metric] = precision_score(
                    y_test, y_pred, average="weighted", zero_division=0
                )
            elif metric == "mcc":
                results[metric] = matthews_corrcoef(y_test, y_pred)
            elif metric == "fpr_micro":
                results[metric] = fpr["micro"]
            elif metric == "fpr_macro":
                results[metric] = fpr["macro"]
            elif metric == "tpr_micro":
                results[metric] = tpr["micro"]
            elif metric == "tpr_macro":
                results[metric] = tpr["macro"]
            else:
                raise ValueError(f"invalid metric {metric}")

        log.debug(f"evaluate_classifier: {results}")
        return results

    def fpr_tpr(
        self,
        y_test: np.ndarray,
        y_pred_proba: np.ndarray,
        classes: list,
    ) -> Tuple[dict, dict]:
        n_classes = len(np.unique(classes))

        if n_classes > 2:
            y_onehot_test = label_binarize(y_test, classes=classes)

            fpr, tpr = dict(), dict()
            fpr_per_class, tpr_per_class = dict(), dict()
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(
                y_onehot_test.ravel(), y_pred_proba.ravel()
            )
            # roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            for i in range(n_classes):
                fpr_per_class[i], tpr_per_class[i], _ = roc_curve(
                    y_onehot_test[:, i], y_pred_proba[:, i]
                )
                # roc_auc[i] = auc(fpr_per_class[i], tpr_per_class[i])

            fpr_grid = np.linspace(0.0, 1.0, 1000)

            # Interpolate all ROC curves at these points
            mean_tpr = np.zeros_like(fpr_grid)

            for i in range(n_classes):
                mean_tpr += np.interp(
                    fpr_grid, fpr_per_class[i], tpr_per_class[i]
                )  # linear interpolation

            # Average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = fpr_grid
            tpr["macro"] = mean_tpr
        else:
            y_onehot_test = label_binarize(y_test, classes=classes)
            fpr, tpr = dict(), dict()
            fpr_per_class, tpr_per_class = dict(), dict()
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(
                y_onehot_test.ravel(), y_pred_proba[:, 1]
            )

            fpr_per_class, tpr_per_class, _ = roc_curve(
                y_onehot_test, y_pred_proba[:, 1]
            )

            fpr_grid = np.linspace(0.0, 1.0, 1000)

            # Interpolate all ROC curves at these points
            mean_tpr = np.zeros_like(fpr_grid)

            mean_tpr = np.interp(
                fpr_grid, fpr_per_class, tpr_per_class
            )  # linear interpolation

            fpr["macro"] = fpr_grid
            tpr["macro"] = mean_tpr

        return fpr, tpr

    def roc_auc_score(
        self,
        y_test: np.ndarray,
        y_pred_proba: np.ndarray,
        classes: list,
        multi_class: str = "ovr",
        average: str = "micro",
    ) -> float:
        return evaluate_auc(
            y_test,
            y_pred_proba,
            classes,
            multi_class=multi_class,
            average=average,
        )[0]

    def average_precision_score(
        self,
        y_test: np.ndarray,
        y_pred_proba: np.ndarray,
        classes: list,
        average: str = "macro",
    ) -> float:
        return evaluate_auc(y_test, y_pred_proba, classes, average=average)[1]


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_classifier(
    estimator: Any,
    X: Union[pd.DataFrame, np.ndarray],
    Y: Union[pd.Series, np.ndarray, List],
    n_folds: int = 3,
    seed: int = 0,
    pretrained: bool = False,
    group_ids: Optional[pd.Series] = None,
    classes: Any = None,
) -> Dict:
    """Helper for evaluating classifiers.

    Args:
        estimator:
            Baseline model to evaluate. if pretrained == False, it must not be fitted.
        X: pd.DataFrame or np.ndarray:
            The covariates
        Y: pd.Series or np.ndarray or list:
            The labels
        n_folds: int
            cross-validation folds
        seed: int
            Random seed
        pretrained: bool
            If the estimator was already trained or not.
        group_ids: pd.Series
            The group_ids to use for stratified cross-validation

    Returns:
        Dict containing "raw" and "str" nodes. The "str" node contains prettified metrics, while the raw metrics includes tuples of form (`mean`, `std`) for each metric.
        Both "raw" and "str" nodes contain the following metrics:
            - "aucroc_ovo_macro", "aucroc_ovr_macro", "aucroc_ovr_micro" : the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
            - "aucprc" : The average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
            - "accuracy" : Accuracy classification score.
            - "f1_score_micro": F1 score is a harmonic mean of the precision and recall. This version uses the "micro" average: calculate metrics globally by counting the total true positives, false negatives and false positives.
            - "f1_score_macro": F1 score is a harmonic mean of the precision and recall. This version uses the "macro" average: calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
            - "f1_score_weighted": F1 score is a harmonic mean of the precision and recall. This version uses the "weighted" average: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
            - "kappa":  computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.
            - "precision_micro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(micro) calculates metrics globally by counting the total true positives.
            - "precision_macro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
            - "precision_weighted": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
            - "recall_micro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(micro) calculates metrics globally by counting the total true positives.
            - "recall_macro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
            - "recall_weighted": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
            - "mcc": The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.

    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    if classes is None:
        classes = np.ravel(Y)
    classes = set(classes)

    enable_reproducible_results(seed)

    X = pd.DataFrame(X).reset_index(drop=True)
    Y = LabelEncoder().fit_transform(Y)
    Y = pd.Series(Y).reset_index(drop=True)
    if group_ids is not None:
        group_ids = pd.Series(group_ids).reset_index(drop=True)

    log.debug(f"evaluate_estimator shape x:{X.shape} y:{Y.shape}")

    results = {}

    evaluator = classifier_metrics()
    for metric in clf_supported_metrics:
        results[metric] = np.zeros(n_folds)

    indx = 0
    if group_ids is not None:
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # group_ids is always ignored for StratifiedKFold so safe to pass None
    for train_index, test_index in skf.split(X, Y, groups=group_ids):
        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]

        if pretrained:
            model = estimator[indx]
        else:
            model = copy.deepcopy(estimator)
            model.fit(X_train, Y_train)

        preds = model.predict_proba(X_test)
        pred_labels = set(np.ravel(model.predict(X_test)))
        classes = classes.union(pred_labels)

        scores = evaluator.score_proba(Y_test, preds, list(sorted(classes)))

        for metric in scores:
            if "fpr" not in metric and "tpr" not in metric:
                results[metric][indx] = scores[metric]
            else:
                results[metric] = scores[metric]

        indx += 1

    output_clf = {}
    output_clf_str = {}

    for key in results:
        if "fpr" not in key and "tpr" not in key:
            key_out = generate_score(results[key])
            output_clf[key] = key_out
            output_clf_str[key] = print_score(key_out)
        else:
            output_clf[key] = results[key]

    return {
        "raw": output_clf,
        "str": output_clf_str,
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_ts_classifier(
    estimator: Any,
    static_data: Optional[Union[pd.DataFrame, np.ndarray]],
    temporal_data: Union[List, np.ndarray],
    observation_times: Union[List, np.ndarray],
    labels: Union[pd.Series, np.ndarray, List],
    n_folds: int = 3,
    seed: int = 0,
    pretrained: bool = False,
    group_ids: Optional[pd.Series] = None,
    classes: Any = None,
) -> Dict:
    """Helper for evaluating classifiers.

    Args:
        estimator:
            Baseline model to evaluate. if pretrained == False, it must not be fitted.
        X: pd.DataFrame or np.ndarray:
            The covariates
        Y: pd.Series or np.ndarray or list:
            The labels
        n_folds: int
            cross-validation folds
        seed: int
            Random seed
        pretrained: bool
            If the estimator was already trained or not.
        group_ids: pd.Series
            The group_ids to use for stratified cross-validation

    Returns:
        Dict containing "raw" and "str" nodes. The "str" node contains prettified metrics, while the raw metrics includes tuples of form (`mean`, `std`) for each metric.
        Both "raw" and "str" nodes contain the following metrics:
            - "aucroc" : the Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores.
            - "aucprc" : The average precision summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the increase in recall from the previous threshold used as the weight.
            - "accuracy" : Accuracy classification score.
            - "f1_score_micro": F1 score is a harmonic mean of the precision and recall. This version uses the "micro" average: calculate metrics globally by counting the total true positives, false negatives and false positives.
            - "f1_score_macro": F1 score is a harmonic mean of the precision and recall. This version uses the "macro" average: calculate metrics for each label, and find their unweighted mean. This does not take label imbalance into account.
            - "f1_score_weighted": F1 score is a harmonic mean of the precision and recall. This version uses the "weighted" average: Calculate metrics for each label, and find their average weighted by support (the number of true instances for each label).
            - "kappa":  computes Cohen’s kappa, a score that expresses the level of agreement between two annotators on a classification problem.
            - "precision_micro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(micro) calculates metrics globally by counting the total true positives.
            - "precision_macro": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
            - "precision_weighted": Precision is defined as the number of true positives over the number of true positives plus the number of false positives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
            - "recall_micro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(micro) calculates metrics globally by counting the total true positives.
            - "recall_macro": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(macro) calculates metrics for each label, and finds their unweighted mean.
            - "recall_weighted": Recall is defined as the number of true positives over the number of true positives plus the number of false negatives. This version(weighted) calculates metrics for each label, and find their average weighted by support.
            - "mcc": The Matthews correlation coefficient is used in machine learning as a measure of the quality of binary and multiclass classifications. It takes into account true and false positives and negatives and is generally regarded as a balanced measure which can be used even if the classes are of very different sizes.

    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    if classes is None:
        classes = np.ravel(labels)
    classes = set(classes)

    enable_reproducible_results(seed)

    if static_data is None:
        static_data = np.zeros((len(temporal_data), 0))

    temporal_data = np.asarray(temporal_data)
    labels = np.asarray(labels)
    observation_times = np.asarray(observation_times)
    labels = LabelEncoder().fit_transform(labels)

    if group_ids is not None:
        group_ids = np.asarray(group_ids)

    results = {}

    evaluator = classifier_metrics()
    for metric in clf_supported_metrics:
        results[metric] = np.zeros(n_folds)

    indx = 0
    if group_ids is not None:
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    # group_ids is always ignored for StratifiedKFold so safe to pass None
    for train_index, test_index in skf.split(temporal_data, labels, groups=group_ids):
        static_train = static_data[train_index]
        temporal_train = temporal_data[train_index]
        horizons_train = observation_times[train_index]
        labels_train = labels[train_index]

        static_test = static_data[test_index]
        temporal_test = temporal_data[test_index]
        horizons_test = observation_times[test_index]

        labels_test = labels[test_index]

        if pretrained:
            model = estimator[indx]
        else:
            model = copy.deepcopy(estimator)
            model.fit(static_train, temporal_train, horizons_train, labels_train)

        preds = model.predict_proba(static_test, temporal_test, horizons_test)
        pred_labels = set(
            np.ravel(model.predict(static_test, temporal_test, horizons_test))
        )
        classes = classes.union(pred_labels)

        scores = evaluator.score_proba(labels_test, preds, list(sorted(classes)))
        for metric in scores:
            if "fpr" not in metric and "tpr" not in metric:
                results[metric][indx] = scores[metric]
            else:
                results[metric] = scores[metric]

        indx += 1

    output_clf = {}
    output_clf_str = {}

    for key in results:
        if "fpr" not in key and "tpr" not in key:
            key_out = generate_score(results[key])
            output_clf[key] = key_out
            output_clf_str[key] = print_score(key_out)
        else:
            output_clf[key] = results[key]

    return {
        "raw": output_clf,
        "str": output_clf_str,
    }


@validate_arguments(config=dict(arbitrary_types_allowed=True))
def evaluate_regression(
    estimator: Any,
    X: Union[pd.DataFrame, np.ndarray],
    Y: Union[pd.Series, np.ndarray, List],
    n_folds: int = 3,
    seed: int = 0,
    pretrained: bool = False,
    group_ids: Optional[pd.Series] = None,
) -> Dict:
    """Helper for evaluating regression tasks.

    Args:
        estimator:
            Baseline model to evaluate. if pretrained == False, it must not be fitted.
        X: pd.DataFrame or np.ndarray
            covariates
        Y: pd.Series or np.ndarray or list
            outcomes
        n_folds: int
            Number of cross-validation folds
        seed: int
            Random seed
        group_ids: pd.Series
            Optional group_ids for stratified cross-validation

    Returns:
        Dict containing "raw" and "str" nodes. The "str" node contains prettified metrics, while the raw metrics includes tuples of form (`mean`, `std`) for each metric.
        Both "raw" and "str" nodes contain the following metrics:
            - "r2": R^2(coefficient of determination) regression score function.
            - "mse": Mean squared error regression loss.
            - "mae": Mean absolute error regression loss.
    """
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")

    enable_reproducible_results(seed)
    metrics = reg_supported_metrics

    X = pd.DataFrame(X).reset_index(drop=True)
    Y = pd.Series(Y).reset_index(drop=True)
    if group_ids is not None:
        group_ids = pd.Series(group_ids).reset_index(drop=True)

    log.debug(f"evaluate_estimator shape x:{X.shape} y:{Y.shape}")

    metrics_ = {}
    for metric in metrics:
        metrics_[metric] = np.zeros(n_folds)

    indx = 0
    if group_ids is not None:
        kf = GroupKFold(n_splits=n_folds)
    else:
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)

    for train_index, test_index in kf.split(X, Y, groups=group_ids):

        X_train = X.loc[X.index[train_index]]
        Y_train = Y.loc[Y.index[train_index]]
        X_test = X.loc[X.index[test_index]]
        Y_test = Y.loc[Y.index[test_index]]

        if pretrained:
            model = estimator[indx]
        else:
            model = copy.deepcopy(estimator)
            model.fit(X_train, Y_train)

        preds = model.predict(X_test)

        metrics_["mse"][indx] = mean_squared_error(Y_test, preds)
        metrics_["mae"][indx] = mean_absolute_error(Y_test, preds)
        metrics_["r2"][indx] = r2_score(Y_test, preds)

        indx += 1

    output_mse = generate_score(metrics_["mse"])
    output_mae = generate_score(metrics_["mae"])
    output_r2 = generate_score(metrics_["r2"])

    return {
        "raw": {
            "mse": output_mse,
            "mae": output_mae,
            "r2": output_r2,
        },
        "str": {
            "mse": print_score(output_mse),
            "mae": print_score(output_mae),
            "r2": print_score(output_r2),
        },
    }


def generate_score(metric: np.ndarray) -> Tuple[float, float]:
    percentile_val = 1.96
    return (np.mean(metric), percentile_val * np.std(metric) / np.sqrt(len(metric)))


def print_score(score: Tuple[float, float]) -> str:
    return str(round(score[0], 4)) + " +/- " + str(round(score[1], 3))
