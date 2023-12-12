# stdlib
from typing import Any, Callable, List, Optional, Tuple, Union

# third party
import numpy as np
import torch
from pydantic import validate_arguments
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, sampler
from tqdm import tqdm
from tsai.models.InceptionTime import InceptionTime
from tsai.models.InceptionTimePlus import InceptionTimePlus
from tsai.models.OmniScaleCNN import OmniScaleCNN
from tsai.models.ResCNN import ResCNN
from tsai.models.RNN_FCN import MLSTM_FCN
from tsai.models.TCN import TCN
from tsai.models.TransformerModel import TransformerModel
from tsai.models.XceptionTime import XceptionTime
from tsai.models.XCM import XCM

# tls_fingerprinting absolute
import tls_fingerprinting.logger as log
from tls_fingerprinting.models.base.nn.mlp import (
    MLP,
    GumbelSoftmax,
    MultiActivationHead,
    get_nonlin,
)
from tls_fingerprinting.utils.constants import DEVICE
from tls_fingerprinting.utils.reproducibility import enable_reproducible_results

# tls_fingerprinting relative
# tls_crawler relative
from .samplers import ImbalancedDatasetSampler

modes = [
    "LSTM",
    "GRU",
    "RNN",
    "Transformer",
    "MLSTM_FCN",
    "TCN",
    "InceptionTime",
    "InceptionTimePlus",
    "XceptionTime",
    "ResCNN",
    "OmniScaleCNN",
    "XCM",
]


class TimeSeriesModel(nn.Module):
    """Basic neural net for time series.

    Args
        task_type: str,
            The type of the problem. Available options: regression, classification
        n_static_units_in: int
            Number of input units for the statis data.
        n_temporal_units_in: int
            Number of units for the temporal features
        n_temporal_window: int,
            Number of temporal observations for each subject
        output_shape: List[int],
            Shape of the output tensor
        n_static_units_hidden: int. Default = 102
            Number of hidden units for the static features
        n_static_layers_hidden: int. Default = 2
            Number of hidden layers for the static features
        n_temporal_units_hidden: int. Default = 100
            Number of hidden units for the temporal features
        n_temporal_layers_hidden: int. Default = 2
            Number of hidden layers for the temporal features
        n_iter: int. Default = 500
            Number of epochs
        mode: str. Default = "RNN"
            Core neural net architecture.
            Available models:
                - "LSTM"
                - "GRU"
                - "RNN"
                - "Transformer"
                - "MLSTM_FCN"
                - "TCN"
                - "InceptionTime"
                - "InceptionTimePlus"
                - "XceptionTime"
                - "ResCNN"
                - "OmniScaleCNN"
                - "XCM"
        n_iter_print: int. Default = 10
            Number of epochs to print the loss.
        batch_size: int. Default = 100
            Batch size
        lr: float. Default = 1e-3
            Learning rate
        weight_decay: float. Default = 1e-3
            l2 (ridge) penalty for the weights.
        window_size: int = 1
            How many hidden states to use for the outcome.
        device: Any = DEVICE
            PyTorch device to use.
        dataloader_sampler: Optional[sampler.Sampler] = None
            Custom data sampler for training.
        nonlin_out: Optional[List[Tuple[str, int]]] = None
            List of activations for the output. Example [("tanh", 1), ("softmax", 3)] - means the output layer will apply "tanh" for the first unit, and softmax for the following 3 units in the output.
        loss: Optional[Callable] = None
            Custom additional loss.
        dropout: float. Default = 0
            Dropout value.
        nonlin: Optional[str]. Default = "relu"
            Activation for hidden layers.
        random_state: int = 0
            Random seed
        clipping_value: int. Default = 1,
            Gradients clipping value. Zero disables the feature
        patience: int. Default = 20
            How many epoch * n_iter_print to wait without loss improvement.
        train_ratio: float = 0.8
            Train/test split ratio
        use_horizon_condition: bool = True
            Whether to predict using the observation times(True) or just the covariates(False).
    """

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        task_type: str,  # regression, classification
        n_static_units_in: int,
        n_temporal_units_in: int,
        n_temporal_window: int,
        output_shape: List[int],
        n_static_units_hidden: int = 102,
        n_static_layers_hidden: int = 1,
        n_temporal_units_hidden: int = 102,
        n_temporal_layers_hidden: int = 1,
        n_iter: int = 500,
        mode: str = "RNN",
        n_iter_print: int = 10,
        batch_size: int = 100,
        lr: float = 1e-3,
        weight_decay: float = 1e-3,
        window_size: int = 1,
        device: Any = DEVICE,
        dataloader_sampler: Optional[sampler.Sampler] = None,
        nonlin_out: Optional[List[Tuple[str, int]]] = None,
        loss: Optional[Callable] = None,
        dropout: float = 0,
        nonlin: Optional[str] = "relu",
        random_state: int = 0,
        clipping_value: int = 1,
        patience: int = 5,
        train_ratio: float = 0.8,
        use_horizon_condition: bool = False,
    ) -> None:
        super(TimeSeriesModel, self).__init__()

        enable_reproducible_results(random_state)

        if task_type not in ["classification", "regression"]:
            raise ValueError(f"Invalid task type {task_type}")
        if mode not in modes:
            raise ValueError(f"Unsupported mode {mode}. Available: {modes}")
        if len(output_shape) == 0:
            raise ValueError("Invalid output shape")

        self.task_type = task_type

        if loss is not None:
            self.loss = loss
        elif task_type == "regression":
            self.loss = nn.MSELoss()
        elif task_type == "classification":
            self.loss = nn.CrossEntropyLoss()

        self.n_iter = n_iter
        self.n_iter_print = n_iter_print
        self.batch_size = batch_size
        self.n_static_units_in = n_static_units_in
        self.n_temporal_units_in = n_temporal_units_in
        self.n_temporal_window = n_temporal_window
        self.n_static_units_hidden = n_static_units_hidden
        self.n_temporal_units_hidden = n_temporal_units_hidden
        self.n_static_layers_hidden = n_static_layers_hidden
        self.n_temporal_layers_hidden = n_temporal_layers_hidden
        self.device = device
        self.window_size = window_size
        self.dataloader_sampler = dataloader_sampler
        self.lr = lr
        self.output_shape = output_shape
        self.n_units_out = np.prod(self.output_shape)
        self.clipping_value = clipping_value
        self.use_horizon_condition = use_horizon_condition

        self.patience = patience
        self.train_ratio = train_ratio
        self.random_state = random_state

        self.temporal_layer = TimeSeriesLayer(
            n_static_units_in=n_static_units_in,
            n_temporal_units_in=n_temporal_units_in
            + int(use_horizon_condition),  # measurements + horizon
            n_temporal_window=n_temporal_window,
            n_units_out=self.n_units_out,
            n_static_units_hidden=n_static_units_hidden,
            n_static_layers_hidden=n_static_layers_hidden,
            n_temporal_units_hidden=n_temporal_units_hidden,
            n_temporal_layers_hidden=n_temporal_layers_hidden,
            mode=mode,
            window_size=window_size,
            device=device,
            dropout=dropout,
            nonlin=nonlin,
            random_state=random_state,
        ).to(device)

        self.mode = mode

        self.out_activation: Optional[nn.Module] = None
        self.n_act_out: Optional[int] = None

        if nonlin_out is not None:
            self.n_act_out = 0
            activations = []
            for nonlin, nonlin_len in nonlin_out:
                self.n_act_out += nonlin_len
                activations.append((get_nonlin(nonlin), nonlin_len))

            if self.n_units_out % self.n_act_out != 0:
                raise RuntimeError(
                    f"Shape mismatch for the output layer. Expected length {self.n_units_out}, but got {nonlin_out} with length {self.n_act_out}"
                )
            self.out_activation = MultiActivationHead(activations, device=device)
        elif self.task_type == "classification":
            self.n_act_out = self.n_units_out
            self.out_activation = GumbelSoftmax().to(device)

        self.optimizer = torch.optim.Adam(
            self.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
        )  # optimize all rnn parameters

    def to(self, device: Any) -> Any:
        self.device = device
        self.temporal_layer.to(device)

        return self

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        observation_times: torch.Tensor,
        with_act: bool = True,
    ) -> torch.Tensor:
        # x shape (batch, time_step, input_size)
        # r_out shape (batch, time_step, output_size)

        if torch.isnan(static_data).sum() != 0:
            raise ValueError("NaNs detected in the static data")
        if torch.isnan(temporal_data).sum() != 0:
            raise ValueError("NaNs detected in the temporal data")
        if torch.isnan(observation_times).sum() != 0:
            raise ValueError("NaNs detected in the temporal horizons")

        if self.use_horizon_condition:
            temporal_data_merged = torch.cat(
                [temporal_data, observation_times.unsqueeze(2)], dim=2
            )
        else:
            temporal_data_merged = temporal_data

        if torch.isnan(temporal_data_merged).sum() != 0:
            raise ValueError("NaNs detected in the temporal merged data")

        pred = self.temporal_layer(static_data, temporal_data_merged)

        if self.out_activation is not None:
            pred = pred.reshape(-1, self.n_act_out)
            if with_act:
                pred = self.out_activation(pred)

        pred = pred.reshape(-1, *self.output_shape)

        return pred

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict(
        self,
        static_data: Union[List, np.ndarray],
        temporal_data: Union[List, np.ndarray],
        observation_times: Union[List, np.ndarray],
    ) -> np.ndarray:
        self.eval()
        with torch.no_grad():
            (
                static_data_t,
                temporal_data_t,
                observation_times_t,
                _,
            ) = self._prepare_input(static_data, temporal_data, observation_times)

            yt = self(
                static_data_t,
                temporal_data_t,
                observation_times_t,
                with_act=True,
            )

            if self.task_type == "classification":
                return np.argmax(yt.cpu().numpy(), -1)
            else:
                return yt.cpu().numpy()

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def predict_proba(
        self,
        static_data: Union[List, np.ndarray],
        temporal_data: Union[List, np.ndarray],
        observation_times: Union[List, np.ndarray],
    ) -> np.ndarray:
        assert self.task_type == "classification"
        self.eval()
        with torch.no_grad():
            (
                static_data_t,
                temporal_data_t,
                observation_times_t,
                _,
            ) = self._prepare_input(static_data, temporal_data, observation_times)

            yt = self(
                static_data_t,
                temporal_data_t,
                observation_times_t,
                with_act=True,
            )

            return yt.cpu().numpy()

    def score(
        self,
        static_data: Union[List, np.ndarray],
        temporal_data: Union[List, np.ndarray],
        observation_times: Union[List, np.ndarray],
        outcome: np.ndarray,
    ) -> float:
        y_pred = self.predict(static_data, temporal_data, observation_times)
        if self.task_type == "classification":
            return np.mean(y_pred == outcome)
        else:
            return np.mean(np.inner(outcome - y_pred, outcome - y_pred) / 2.0)

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def fit(
        self,
        static_data: Union[List, np.ndarray],
        temporal_data: Union[List, np.ndarray],
        observation_times: Union[List, np.ndarray],
        outcome: Union[List, np.ndarray],
    ) -> Any:
        (
            static_data_t,
            temporal_data_t,
            observation_times_t,
            outcome_t,
        ) = self._prepare_input(static_data, temporal_data, observation_times, outcome)

        return self._train(
            static_data_t, temporal_data_t, observation_times_t, outcome_t
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def _train(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        observation_times: torch.Tensor,
        outcome: torch.Tensor,
    ) -> Any:
        patience = 0
        prev_error = np.inf

        train_dl, test_dl = self.dataloader(
            static_data,
            temporal_data,
            observation_times,
            outcome,
        )

        # training and testing
        for it in tqdm(range(self.n_iter)):
            train_loss = self._train_epoch(train_dl)
            if it % self.n_iter_print == 0:
                val_loss = self._test_epoch(test_dl)
                log.info(
                    f"Epoch:{it}| train loss: {train_loss}, validation loss: {val_loss}"
                )
                if val_loss < prev_error:
                    patience = 0
                    prev_error = val_loss
                else:
                    patience += 1
                if patience > self.patience:
                    break

        return self

    def _train_epoch(self, loader: DataLoader) -> float:
        self.train()

        losses = []
        for step, (static_mb, temporal_mb, horizons_mb, y_mb) in enumerate(loader):
            self.optimizer.zero_grad()  # clear gradients for this training step

            pred = self(
                static_mb, temporal_mb, horizons_mb, with_act=False
            )  # rnn output
            loss = self.loss(pred, y_mb)

            assert not torch.isnan(loss)

            loss.backward()  # backpropagation, compute gradients
            if self.clipping_value > 0:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clipping_value)
            self.optimizer.step()  # apply gradients

            losses.append(loss.detach().cpu())

        return np.mean(losses)

    def _test_epoch(self, loader: DataLoader) -> float:
        self.eval()

        losses = []
        for step, (static_mb, temporal_mb, horizons_mb, y_mb) in enumerate(loader):
            pred = self(static_mb, temporal_mb, horizons_mb)  # rnn output
            loss = self.loss(pred.squeeze(), y_mb.squeeze())

            losses.append(loss.detach().cpu())

        return np.mean(losses)

    def dataloader(
        self,
        static_data: torch.Tensor,
        temporal_data: torch.Tensor,
        observation_times: torch.Tensor,
        outcome: torch.Tensor,
    ) -> DataLoader:
        stratify = None
        _, out_counts = torch.unique(outcome, return_counts=True)
        if out_counts.min() > 1:
            stratify = outcome.cpu()

        (
            static_data_train,
            static_data_test,
            temporal_data_train,
            temporal_data_test,
            observation_times_train,
            observation_times_test,
            outcome_train,
            outcome_test,
        ) = train_test_split(
            static_data.cpu(),
            temporal_data.cpu(),
            observation_times.cpu(),
            outcome.cpu(),
            train_size=self.train_ratio,
            random_state=self.random_state,
            stratify=stratify,
        )
        train_dataset = TensorDataset(
            static_data_train.to(self.device),
            temporal_data_train.to(self.device),
            observation_times_train.to(self.device),
            outcome_train.to(self.device),
        )
        test_dataset = TensorDataset(
            static_data_test.to(self.device),
            temporal_data_test.to(self.device),
            observation_times_test.to(self.device),
            outcome_test.to(self.device),
        )

        sampler = self.dataloader_sampler
        if sampler is None and self.task_type == "classification":
            sampler = ImbalancedDatasetSampler(
                list(np.asarray(outcome_train).squeeze()),
            )

        return (
            DataLoader(
                train_dataset,
                batch_size=self.batch_size,
                sampler=sampler,
                pin_memory=False,
            ),
            DataLoader(
                test_dataset,
                batch_size=self.batch_size,
                pin_memory=False,
            ),
        )

    def _check_tensor(self, X: torch.Tensor) -> torch.Tensor:
        if isinstance(X, torch.Tensor):
            return X.to(self.device)
        else:
            return torch.from_numpy(np.asarray(X)).to(self.device)

    def _prepare_input(
        self,
        static_data: Union[List, np.ndarray],
        temporal_data: Union[List, np.ndarray],
        observation_times: Union[List, np.ndarray],
        outcome: Optional[Union[List, np.ndarray]] = None,
    ) -> Tuple:
        static_data = np.asarray(static_data)
        temporal_data = np.asarray(temporal_data).astype(float)
        observation_times = np.asarray(observation_times).astype(float)
        if outcome is not None:
            outcome = np.asarray(outcome)

        static_data_t = self._check_tensor(static_data).float()

        temporal_data = temporal_data.astype(float)
        temporal_data_t = self._check_tensor(temporal_data).float()
        observation_times_t = self._check_tensor(observation_times).float()

        outcome_t = None
        if outcome is not None:
            outcome_t = self._check_tensor(outcome).float()

            if self.task_type == "classification":
                outcome_t = outcome_t.long()

        return (
            static_data_t,
            temporal_data_t,
            observation_times_t,
            outcome_t,
        )


class TimeSeriesLayer(nn.Module):
    def __init__(
        self,
        n_static_units_in: int,
        n_temporal_units_in: int,
        n_temporal_window: int,
        n_units_out: int,
        n_static_units_hidden: int = 100,
        n_static_layers_hidden: int = 2,
        n_temporal_units_hidden: int = 100,
        n_temporal_layers_hidden: int = 2,
        mode: str = "RNN",
        window_size: int = 1,
        device: Any = DEVICE,
        dropout: float = 0,
        nonlin: Optional[str] = "relu",
        random_state: int = 0,
    ) -> None:
        super(TimeSeriesLayer, self).__init__()
        temporal_params = {
            "input_size": n_temporal_units_in,
            "hidden_size": n_temporal_units_hidden,
            "num_layers": n_temporal_layers_hidden,
            "dropout": 0 if n_temporal_layers_hidden == 1 else dropout,
            "batch_first": True,
        }
        temporal_models = {
            "RNN": nn.RNN,
            "LSTM": nn.LSTM,
            "GRU": nn.GRU,
        }

        if mode in ["RNN", "LSTM", "GRU"]:
            self.temporal_layer = temporal_models[mode](**temporal_params)
        elif mode == "MLSTM_FCN":
            self.temporal_layer = MLSTM_FCN(
                c_in=n_temporal_units_in,
                c_out=n_temporal_units_hidden,
                hidden_size=n_temporal_units_hidden,
                rnn_layers=n_temporal_layers_hidden,
                fc_dropout=dropout,
                seq_len=n_temporal_window,
                shuffle=False,
            )
        elif mode == "TCN":
            self.temporal_layer = TCN(
                c_in=n_temporal_units_in,
                c_out=n_temporal_units_hidden,
                fc_dropout=dropout,
            )
        elif mode == "InceptionTime":
            self.temporal_layer = InceptionTime(
                c_in=n_temporal_units_in,
                c_out=n_temporal_units_hidden,
                depth=n_temporal_layers_hidden,
                seq_len=n_temporal_window,
            )
        elif mode == "InceptionTimePlus":
            self.temporal_layer = InceptionTimePlus(
                c_in=n_temporal_units_in,
                c_out=n_temporal_units_hidden,
                depth=n_temporal_layers_hidden,
                seq_len=n_temporal_window,
            )
        elif mode == "XceptionTime":
            self.temporal_layer = XceptionTime(
                c_in=n_temporal_units_in,
                c_out=n_temporal_units_hidden,
            )
        elif mode == "ResCNN":
            self.temporal_layer = ResCNN(
                c_in=n_temporal_units_in,
                c_out=n_temporal_units_hidden,
            )
        elif mode == "OmniScaleCNN":
            self.temporal_layer = OmniScaleCNN(
                c_in=n_temporal_units_in,
                c_out=n_temporal_units_hidden,
                seq_len=max(n_temporal_window, 10),
            )
        elif mode == "XCM":
            self.temporal_layer = XCM(
                c_in=n_temporal_units_in,
                c_out=n_temporal_units_hidden,
                seq_len=n_temporal_window,
                fc_dropout=dropout,
            )
        elif mode == "Transformer":
            self.temporal_layer = TransformerModel(
                c_in=n_temporal_units_in,
                c_out=n_temporal_units_hidden,
                dropout=dropout,
                n_layers=n_temporal_layers_hidden,
            )
        else:
            raise RuntimeError(f"Unknown TS mode {mode}")

        self.device = device
        self.mode = mode

        if mode in ["RNN", "LSTM", "GRU"]:
            self.out = WindowLinearLayer(
                n_static_units_in=n_static_units_in,
                n_temporal_units_in=n_temporal_units_hidden,
                window_size=window_size,
                n_units_out=n_units_out,
                n_layers=n_static_layers_hidden,
                dropout=dropout,
                nonlin=nonlin,
                device=device,
            )
        else:
            self.out = MLP(
                task_type="regression",
                n_units_in=n_static_units_in + n_temporal_units_hidden,
                n_units_out=n_units_out,
                n_layers_hidden=n_static_layers_hidden,
                n_units_hidden=n_static_units_hidden,
                dropout=dropout,
                nonlin=nonlin,
                device=device,
            )

        self.temporal_layer.to(device)
        self.out.to(device)

    def forward(
        self, static_data: torch.Tensor, temporal_data: torch.Tensor
    ) -> torch.Tensor:
        if self.mode in ["RNN", "LSTM", "GRU"]:
            X_interm, _ = self.temporal_layer(temporal_data)

            if torch.isnan(X_interm).sum() != 0:
                raise RuntimeError("NaNs detected in the temporal embeddings")

            return self.out(static_data, X_interm)
        else:
            X_interm = self.temporal_layer(torch.swapaxes(temporal_data, 1, 2))

            if torch.isnan(X_interm).sum() != 0:
                raise RuntimeError("NaNs detected in the temporal embeddings")

            return self.out(torch.cat([static_data, X_interm], dim=1))


class WindowLinearLayer(nn.Module):
    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def __init__(
        self,
        n_static_units_in: int,
        n_temporal_units_in: int,
        window_size: int,
        n_units_out: int,
        n_units_hidden: int = 100,
        n_layers: int = 1,
        dropout: float = 0,
        nonlin: Optional[str] = "relu",
        device: Any = DEVICE,
    ) -> None:
        super(WindowLinearLayer, self).__init__()

        self.device = device
        self.window_size = window_size
        self.n_static_units_in = n_static_units_in
        self.model = MLP(
            task_type="regression",
            n_units_in=n_static_units_in + n_temporal_units_in * window_size,
            n_units_out=n_units_out,
            n_layers_hidden=n_layers,
            n_units_hidden=n_units_hidden,
            dropout=dropout,
            nonlin=nonlin,
            device=device,
        )

    @validate_arguments(config=dict(arbitrary_types_allowed=True))
    def forward(
        self, static_data: torch.Tensor, temporal_data: torch.Tensor
    ) -> torch.Tensor:
        if self.n_static_units_in > 0 and len(static_data) != len(temporal_data):
            raise ValueError("Length mismatch between static and temporal data")

        batch_size, seq_len, n_feats = temporal_data.shape
        temporal_batch = temporal_data[:, seq_len - self.window_size :, :].reshape(
            batch_size, n_feats * self.window_size
        )
        batch = torch.cat([static_data, temporal_batch], axis=1)

        return self.model(batch).to(self.device)
