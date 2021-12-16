# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np
from sklearn.inspection import plot_partial_dependence
from sklearn.base import BaseEstimator


class _model_wrapper(BaseEstimator):
    def __init__(self, base_torch_model, n_monotonic_features):
        """Instantiates sklearn's model wraping torch's model.

        Args:
            base_torch_model (torch.nn.Module): Torch model to be wrapped.
            n_monotonic_features (int): Number of monotonic features.
        """
        super().__init__()
        self._estimator_type = "regressor"  # This is just to avoid sklearn throwing ValueError when the model is used.
        base_torch_model.eval()
        self.device = next(base_torch_model.parameters()).device
        self.base_model = base_torch_model
        self.n_monotonic_features = n_monotonic_features

    def predict(self, X):
        """Predicts from data X using the base model.

        If the outputs of the model are multi-dimensional, we assume a multi-way classifier and use the 
            gap between the top-2 logits as the output to plot against.

        Args:
            X (torch.FloatTensor): Feature matrix with shape [BATCH_SIZE, FEATURE_DIMENSION].

        Returns:
            np.array: Model predictions from X.
        """
        X = torch.from_numpy(X).to(self.device)
        with torch.no_grad():
            predictions = self.base_model(
                X[:, : self.n_monotonic_features], X[:, self.n_monotonic_features :]
            )
            if predictions.shape[-1] > 2:  # k-way classifiers
                predictions, _ = predictions.max(
                    dim=-1
                )  # Logit corresponding to prediction.

        return predictions.detach().cpu().numpy()

    def fit(self):
        self.fitted_ = True
        return self  # Dummy method simply used to force wrapper to be treated as a fit predictor.


def plot_pdp(model, data, n_monotonic_features, out_path="./pdp.jpg"):
    """Saves partial dependence plots.

    This function uses sklearn's implementation of PDP:
    https://scikit-learn.org/stable/modules/partial_dependence.html#partial-dependence

    Args:
        model (torch.nn.Module): Model to be evaluated.
        data (torch.FloatTensor): Feature matrix.
        n_monotonic_features (int): Number of features with respect to which we expect to be monotonic.
            Indices of monotonic features should be [0:n_monotonic_features].
        out_path (str, optional): Path to save plot. Defaults to "./pdp.jpg".
    """
    data = data.cpu().numpy()

    # 'fit()' needs to be called since following methods require a trained predictor.
    # The call to 'fit()' in this case however has no effect on the model.
    sklearn_model = _model_wrapper(model, n_monotonic_features).fit()

    ppd = plot_partial_dependence(
        sklearn_model, data, list(range(max(n_monotonic_features, data.shape[-1])))
    )

    ppd.figure_.savefig(out_path)


def compute_l1_norm(model):
    """Computes L1 norm of parameters of linear layers within model.

    Args:
        model (torch.nn.Module): Model to be evaluated.

    Returns:
        torch.FloatTensor: L1 norm of parameters of linear layers.
    """

    linear_layers_params = []
    for module in model.modules():
        if isinstance(module, torch.nn.Sequential):
            for sub_module in module:
                if isinstance(sub_module, torch.nn.Linear):
                    for p in sub_module.parameters():
                        linear_layers_params.append(p.view(-1))
        else:
            if isinstance(module, torch.nn.Linear):
                for p in sub_module.parameters():
                    linear_layers_params.append(p.view(-1))

    linear_layers_params = torch.cat(linear_layers_params)
    l1_norm = torch.norm(linear_layers_params, 1)
    return l1_norm


def make_confidence_interval(data):
    """Computes 95% confidence interval given list of values.

    Args:
        data (list): List of values for which a confidence interval
            is to be estimated.

    Returns:
        tuple: Center and extrema of 95% confidence interval.
    """
    data_mean, data_ci95 = (
        np.mean(data),
        1.96 * np.std(data) / np.sqrt(float(len(data))),
    )

    return data_mean, data_ci95

