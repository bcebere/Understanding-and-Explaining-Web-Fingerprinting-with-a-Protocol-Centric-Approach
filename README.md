# Understanding-and-Explaining-Web-Fingerprinting-with-a-Data-Centric-Approach

<p align="center">
  <img src="https://github.com/bcebere/Understanding-and-Explaining-Web-Fingerprinting-with-a-Data-Centric-Approach/assets/1623754/a65ae505-6c7d-4750-a3bc-d3c1d3c69e36"/>
</p>

In this repository, we provide the code to reproduce the results in the "Understanding-and-Explaining-Web-Fingerprinting-with-a-Data-Centric-Approach" paper.  

## Repository structure
This repository is organized as follows:
```bash
models/
    |- src/                                 # Models and benchmarks
    |- tests/                               # Unit tests for the models 
```

## Models

The evaluation models are located in the `models`, and they are organized in a standalone library.
The library can be installed using
```bash
cd models/
pip install -e .
pip install -e .[testing] # for the development setup
```

Usage examples:

```python
# XGBoost usage and evaluation example
# third party
from sklearn.datasets import load_iris

# tls_fingerprinting absolute
from tls_fingerprinting.models.base.static.xgb import XGBoostClassifier as model
from tls_fingerprinting.utils.evaluation import evaluate_classifier

test_plugin = model()
X, y = load_iris(return_X_y=True, as_frame=True)
scores = evaluate_classifier(test_plugin, X, y)
print(scores["str"])

# Example Output
# {'aucroc_ovo_macro': '0.9832 +/- 0.004', 'aucroc_ovr_micro': '0.9841 +/- 0.008', 'aucroc_ovr_weighted': '0.9832 +/- 0.004', 'aucprc_weighted': '0.9766 +/- 0.005', 'aucprc_macro': '0.9766 +/- 0.005', 'aucprc_micro': '0.9766 +/- 0.005', 'accuracy': '0.9333 +/- 0.021', 'f1_score_micro': '0.9333 +/- 0.021', 'f1_score_macro': '0.933 +/- 0.021', 'f1_score_weighted': '0.9331 +/- 0.022', 'kappa': '0.9 +/- 0.032', 'kappa_quadratic': '0.9501 +/- 0.016', 'precision_micro': '0.9333 +/- 0.021', 'precision_macro': '0.933 +/- 0.021', 'precision_weighted': '0.9333 +/- 0.021', 'recall_micro': '0.9333 +/- 0.021', 'recall_macro': '0.9334 +/- 0.021', 'recall_weighted': '0.9333 +/- 0.021', 'mcc': '0.9002 +/- 0.032'}

```

M
