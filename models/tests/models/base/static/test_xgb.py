# third party
from sklearn.datasets import load_iris

# tls_fingerprinting absolute
from tls_fingerprinting.models.base.static.xgb import XGBoostClassifier as plugin
from tls_fingerprinting.utils.evaluation import evaluate_classifier


def test_xgboost_plugin_sanity() -> None:
    test_plugin = plugin()
    assert test_plugin is not None


def test_xgboost_plugin_fit_predict_proba() -> None:
    test_plugin = plugin()
    X, y = load_iris(return_X_y=True, as_frame=True)
    scores = evaluate_classifier(test_plugin, X, y)
    print(scores["str"])
