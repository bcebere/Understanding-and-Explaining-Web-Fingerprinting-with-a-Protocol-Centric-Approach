# stdlib
from typing import Any

# third party
import numpy as np
import pytest

# tls_fingerprinting absolute
# synthcity absolute
from tls_fingerprinting.models.base.nn.ts_vae import TimeSeriesVAE
from tls_fingerprinting.utils.datasets.time_series.google_stocks import (
    GoogleStocksDataloader,
)
from tls_fingerprinting.utils.datasets.time_series.sine import SineDataloader


def test_network_config() -> None:
    static, temporal, observation_times, _ = SineDataloader().load()
    net = TimeSeriesVAE(
        n_static_units=static.shape[-1],
        n_static_units_embedding=static.shape[-1],
        n_temporal_units=temporal[0].shape[-1],
        n_temporal_window=len(temporal[0]),
        n_temporal_units_embedding=temporal[0].shape[-1],
        # decoder
        decoder_n_layers_hidden=2,
        decoder_n_units_hidden=100,
        decoder_nonlin="elu",
        decoder_batch_norm=False,
        decoder_dropout=0,
        # encoder
        encoder_n_layers_hidden=3,
        encoder_n_units_hidden=100,
        encoder_nonlin="elu",
        encoder_batch_norm=False,
        encoder_dropout=0,
        # Training
        weight_decay=1e-3,
        n_iter=1001,
        lr=1e-3,
        batch_size=64,
        n_iter_print=100,
        random_state=77,
        clipping_value=1,
    )

    assert net.batch_size == 64
    assert net.lr == 1e-3
    assert net.n_iter == 1001
    assert net.random_state == 77


@pytest.mark.parametrize("source", [SineDataloader, GoogleStocksDataloader])
def test_ts_vae_generation(source: Any) -> None:
    static, temporal, observation_times, _ = source().load()

    model = TimeSeriesVAE(
        n_static_units=static.shape[-1],
        n_static_units_embedding=static.shape[-1],
        n_temporal_units=temporal[0].shape[-1],
        n_temporal_window=len(temporal[0]),
        n_temporal_units_embedding=temporal[0].shape[-1],
        n_iter=10,
    )
    model.fit(static, temporal, observation_times)

    static_gen, temporal_gen, observation_times_gen = model.generate(10)

    assert static_gen.shape == (10, static.shape[1])
    assert len(observation_times_gen) == len(temporal_gen)
    assert np.asarray(temporal_gen).shape == (
        10,
        temporal[0].shape[0],
        temporal[0].shape[1],
    )
