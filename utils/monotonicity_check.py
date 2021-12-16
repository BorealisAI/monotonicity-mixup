# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
from utils.mixup_utils import interpolate_data_batch
import random


def monotonicity_check(
    model,
    number_of_points,
    n_monotonic_features,
    n_features=-1,
    data_x=None,
    mixup=False,
):
    """Computes fraction of number_of_points where monotonicity IS NOT verified.

    Args:
        model (torch.nn.module): Maps data to the output space.
        number_of_points (int): Maximum number of input points/pairs to be considered.
        n_monotonic_features (int): Number of features with respect to which we expect to be monotonic.
            Indices of monotonic features should be [0:n_monotonic_features].
        n_features (int, optional): Dimension of input space. Required if data_x is None. Defaults to -1.
        data_x (torch.FloatTensor, optional): Data to be used for monotonicity check. Defaults to None,
            in which case data randomly drawn uniformly across the input space will be used.
        mixup (bool, optional): Whether to mixup data_x prior to checking monotonicity. Defaults to False.

    Returns:
        float: number_of_non_monotonic_points / number_of_points.
    """

    model.eval()  # Set model to eval mode.
    device = next(model.parameters()).device

    if data_x is None:
        assert (
            n_features >= n_monotonic_features
        ), "n_features is required if data_x is None."
        data_x = torch.rand(number_of_points, n_features)
    else:
        if mixup:
            # Get interpolated data.
            data_x = interpolate_data_batch(data_x, number_of_points)
        else:
            # Ensure the budget is respected.
            if number_of_points < data_x.shape[0]:
                idx = random.sample(list(range(data_x.shape[0])), number_of_points)
                data_x = data_x[idx, :]

    data_x = data_x.to(device)
    data_monotonic = data_x[:, :n_monotonic_features]
    data_non_monotonic = data_x[:, n_monotonic_features:]

    data_monotonic.requires_grad = (
        True  # Required since we need to compute grad w.r.t. inputs
    )

    # Predictions are computed to enable gradient computation.
    predictions = model(data_monotonic, data_non_monotonic,)

    grad_wrt_monotonic_input = torch.autograd.grad(
        torch.sum(
            predictions.max(dim=1)[0]
        ),  # Get gradients w.r.t. max logit if predictions.ndim>2.
        data_monotonic,  # Only monotonic features are accounted for.
        create_graph=True,
        allow_unused=True,
    )[0]

    # Get component with minimum gradient.
    min_grad_wrt_monotonic_input = grad_wrt_monotonic_input.min(1)[0]
    # Count the points where gradients are negative.
    number_of_non_monotonic_points = float(
        (min_grad_wrt_monotonic_input < 0.0).sum().item()
    )

    return number_of_non_monotonic_points / data_x.shape[0]
