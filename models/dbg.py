# XGBoost usage example
# third party
from sklearn.datasets import load_iris

# tls_fingerprinting absolute
from tls_fingerprinting.models.base.static.xgb import XGBoostClassifier as model
from tls_fingerprinting.utils.evaluation import evaluate_classifier

test_plugin = model()
X, y = load_iris(return_X_y=True, as_frame=True)
scores = evaluate_classifier(test_plugin, X, y)
print(scores["str"])
