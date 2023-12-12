# third party
import numpy as np
import pytest
from sklearn.datasets import load_diabetes, load_digits
from sklearn.preprocessing import MinMaxScaler

# tls_fingerprinting absolute
from tls_fingerprinting.models.base.nn.vae_mlp import VAEMLP


def test_network_config() -> None:
    net = VAEMLP(
        task_type="regression",
        n_features=10,
        n_units_embedding=2,
        n_units_out=3,
        # decoder
        decoder_n_layers_hidden=2,
        decoder_n_units_hidden=100,
        decoder_nonlin="elu",
        decoder_nonlin_out=[("sigmoid", 10)],
        decoder_batch_norm=False,
        decoder_dropout=0,
        # encoder
        encoder_n_layers_hidden=3,
        encoder_n_units_hidden=100,
        encoder_nonlin="elu",
        encoder_batch_norm=False,
        encoder_dropout=0,
        # output
        output_n_layers_hidden=3,
        output_n_units_hidden=100,
        output_nonlin="elu",
        output_batch_norm=False,
        output_dropout=0,
        # Training
        n_iter=1001,
        lr=1e-3,
        weight_decay=1e-3,
        batch_size=64,
        n_iter_print=100,
        random_state=77,
        clipping_value=1,
    )

    assert net.vae is not None
    assert net.prediction is not None
    assert net.task_type == "regression"


@pytest.mark.parametrize("nonlin", ["relu", "elu", "leaky_relu"])
@pytest.mark.parametrize("dropout", [0, 0.5, 0.2])
@pytest.mark.parametrize("batch_norm", [True, False])
@pytest.mark.parametrize("lr", [1e-3, 3e-4])
@pytest.mark.parametrize("hidden", [2, 3])
def test_basic_network(
    nonlin: str,
    dropout: float,
    batch_norm: bool,
    lr: float,
    hidden: int,
) -> None:
    net = VAEMLP(
        task_type="classification",
        n_features=10,
        n_units_embedding=2,
        n_units_out=2,
        n_iter=10,
        lr=lr,
        decoder_dropout=dropout,
        encoder_dropout=dropout,
        output_dropout=dropout,
        decoder_nonlin=nonlin,
        encoder_nonlin=nonlin,
        output_nonlin=nonlin,
        decoder_batch_norm=batch_norm,
        encoder_batch_norm=batch_norm,
        output_batch_norm=batch_norm,
        decoder_n_layers_hidden=hidden,
        encoder_n_layers_hidden=hidden,
        output_n_layers_hidden=hidden,
    )

    assert net.vae.n_iter == 10
    assert net.prediction.n_iter == 10
    assert net.vae.lr == lr
    assert net.prediction.lr == lr


def test_vae_classification() -> None:
    X, y = load_digits(return_X_y=True)
    X = MinMaxScaler().fit_transform(X)

    model = VAEMLP(
        task_type="classification",
        n_features=X.shape[1],
        n_units_embedding=50,
        n_units_out=len(np.unique(y)),
        n_iter=10,
    )
    model.fit(X, y)

    assert model.predict(X).shape == y.shape
    assert model.predict_proba(X).shape == (len(y), 10)


def test_vae_regression() -> None:
    X, y = load_diabetes(return_X_y=True)
    X = MinMaxScaler().fit_transform(X)

    model = VAEMLP(
        task_type="regression",
        n_features=X.shape[1],
        n_units_embedding=50,
        n_units_out=1,
        n_iter=10,
    )

    model.fit(X, y)

    assert model.predict(X).shape == y.shape
    with pytest.raises(ValueError):
        model.predict_proba(X)
