# Understanding-and-Explaining-Web-Fingerprinting-with-a-Data-Centric-Approach

<p align="center">
  <img src="https://github.com/bcebere/Understanding-and-Explaining-Web-Fingerprinting-with-a-Data-Centric-Approach/assets/1623754/a65ae505-6c7d-4750-a3bc-d3c1d3c69e36"/>
</p>

In this repository, we provide the code to reproduce the results in the "Understanding-and-Explaining-Web-Fingerprinting-with-a-Data-Centric-Approach" paper.  
The repository includes reference Machine Learning models for evaluation in the `models` folder and tools for generating HTTPS datasets in the `crawlers` folder.

The code was tested using `Linux Mint 21.2 Victoria` and `Python 3.10`.

## Repository structure
This repository is organized as follows:
```bash
models/
    |- src/                                 # Models and evaluation methods
    |- tests/                               # Unit tests for the ML models
crawlers/
    |- src/                                 # Traffic crawling and parsing 
    |- tests/                               # Unit tests for the crawling logic
experiments/
    |- domains_experiments/                 # Domain experiments 
      |- crawler/                           # Domain crawling logic
      |- scripts/                           # Domain fingerprinting evaluation 
    |- page_wiki_experiments/               # Wikipedia experiments 
      |- crawler/                           # Wikipedia crawling logic
      |- scripts/                           # Wikipedia fingerprinting evaluation 
    |- page_9gag_experiments/               # 9GAG experiments 
      |- crawler/                           # Wikipedia crawling logic
      |- scripts/                           # Wikipedia fingerprinting evaluation 
    |- page_imdb_experiments/               # IMDB experiments 
      |- crawler/                           # Wikipedia crawling logic
      |- scripts/                           # Wikipedia fingerprinting evaluation 
```

## Models

The evaluation models are located in the `models`, and they are organized in a standalone library.
The library can be installed using
```bash
cd models/
pip install -e .
pip install -e .[testing] # for the development setup
```

### Usage examples:

XGBoost
```python
# XGBoost usage and evaluation example
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
Neural Nets
```python
# MLP usage and evaluation example
from sklearn.datasets import load_iris
import numpy as np

from tls_fingerprinting.models.base.nn.mlp import MLP as model
from tls_fingerprinting.utils.evaluation import evaluate_classifier

X, y = load_iris(return_X_y=True, as_frame=True)
test_plugin = model(
    task_type="classification",
    n_units_in=X.shape[1],
    n_units_out=len(np.unique(y)),
)
scores = evaluate_classifier(test_plugin, X, y)
print(scores["str"])


# Example Output
# {'aucroc_ovo_macro': '0.9791 +/- 0.012', 'aucroc_ovr_micro': '0.9672 +/- 0.024', 'aucroc_ovr_weighted': '0.9787 +/- 0.013', 'aucprc_weighted': '0.9496 +/- 0.034', 'aucprc_macro': '0.9496 +/- 0.034', 'aucprc_micro': '0.9496 +/- 0.034', 'accuracy': '0.8667 +/- 0.087', 'f1_score_micro': '0.8667 +/- 0.087', 'f1_score_macro': '0.8559 +/- 0.104', 'f1_score_weighted': '0.8555 +/- 0.104', 'kappa': '0.8008 +/- 0.13', 'kappa_quadratic': '0.9081 +/- 0.052', 'precision_micro': '0.8667 +/- 0.087', 'precision_macro': '0.9025 +/- 0.039', 'precision_weighted': '0.9038 +/- 0.036', 'recall_micro': '0.8667 +/- 0.087', 'recall_macro': '0.8685 +/- 0.085', 'recall_weighted': '0.8667 +/- 0.087', 'mcc': '0.8235 +/- 0.098'}

```


### Tests
```bash
pytest -vvsx
```

## Traffic Crawlers
The `crawlers` folder contains scripts for generating and parsing PCAP files from lists of URLS.

### Library Installation
```bash
cd crawlers
pip install -e .
pip install -e .[testing]
```

### Docker Build
The experiments use a custom Selenium Docker image, with additional scripts and features. Tu build the images, run

```bash
cd crawlers/docker
docker build --tag selenium-chrome -f Dockerfile_chrome .  
docker build --tag selenium-firefox -f Dockerfile_firefox .
```

### Usage example
See [dataset crawlers](experiments/domains_experiments/crawler/crawl.py).

### Tests
If the library and docker builds worked, the unit tests should pass
```bash
cd crawlers
pytest -vvsx
```

## Experiments

The experiments are available in the `experiments` folder. Each experiment includes the crawling scripts and the fingerprinting evaluation code. The 9GAG and IMDB are not included in the repository due size.

```bash
experiments/
    |- domains_experiments/                 # Domain experiments 
      |- crawler/                           # Domain crawling logic
      |- scripts/                           # Domain fingerprinting evaluation 
    |- page_wiki_experiments/               # Wikipedia experiments 
      |- crawler/                           # Wikipedia crawling logic
      |- scripts/                           # Wikipedia fingerprinting evaluation 
    |- page_9gag_experiments/               # 9GAG experiments 
      |- crawler/                           # Wikipedia crawling logic
      |- scripts/                           # Wikipedia fingerprinting evaluation 
    |- page_imdb_experiments/               # IMDB experiments 
      |- crawler/                           # Wikipedia crawling logic
      |- scripts/                           # Wikipedia fingerprinting evaluation 
```

