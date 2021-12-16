# Copyright (c) 2021-present, Royal Bank of Canada.

# All rights reserved.

#

# This source code is licensed under the license found in the

# LICENSE file in the root directory of this source tree.

#

import random
from itertools import combinations
import numpy as np
import torch


def get_pairs(data, max_n_pairs=1000):
    """Creates two batches corresponding to pairs of instances in a given batch.

    Args:
        data (torch.FloatTensor): batch of data with shape [batch_size, ...].
        max_n_pairs (int, optional): Maximum number of pairs. Defaults to 1000.

    Returns:
        tuple: two aligned batches corresponding to pairs within 'data'.
    """
    all_pairs = list(combinations(range(len(data)), 2))
    if len(all_pairs) > max_n_pairs:
        all_pairs = random.sample(all_pairs, max_n_pairs)
    all_pairs = torch.LongTensor(all_pairs).to(data.device)

    pairs_left = torch.index_select(data, 0, all_pairs[:, 0])
    pairs_right = torch.index_select(data, 0, all_pairs[:, 1])

    return pairs_left, pairs_right


def interpolate_data_batch(data_batch, max_batch_size, interpolation_range=0.5):
    """Mixup style data interpolation.

    Args:
        data_batch (torch.FloatTensor): batch of data with shape [batch_size, ...].
        max_batch_size (int): maximum number of interpolated examples.
        interpolation_range (float, optional): How far from 0.5 interpolation
            factors can take values from [0.5-interpolation_range, 0.5+interpolation_range].

    Returns:
        torch.FloatTensor: batch of interpolated pairs from data_batch.
    """

    pairs = get_pairs(data_batch, max_n_pairs=max_batch_size)

    # Standard mixup interpolator: random convex combination of pairs
    lower_bound = 0.5 - interpolation_range
    upper_bound = 0.5 + interpolation_range
    interpolation_factors = (
        torch.distributions.uniform.Uniform(lower_bound, upper_bound)
        .rsample(sample_shape=(len(pairs[0]),))
        .to(data_batch.device)
    )
    # Create extra dimensions in the interpolation_factors tensor
    interpolation_factors = interpolation_factors[
        (...,) + (None,) * (data_batch.ndim - 1)
    ]

    # Interpolation for a pair x_0, x_1 and factor t is given by t*(x_0)+(1-t)*x_1
    with torch.no_grad():
        interpolated_batch = (
            interpolation_factors * pairs[0] + (1.0 - interpolation_factors) * pairs[1]
        )

    # Getting rid of any taped interpolation operation
    interpolated_batch = interpolated_batch.detach()

    return interpolated_batch


def compute_mixup_regularizer(
    model, data_x, number_of_pairs, n_monotonic_features, use_random_data=False
):
    """Compute gradient penalty on interpolated data.

    Args:
        model (torch.nn.module): Maps data to the output space.
        data_x (torch.FloatTensor): Batch of input data to be interpolated.
        number_of_pairs (int): Maximum number of input pairs to be considered.
        n_monotonic_features (int): Number of features with respect to which we expect to be monotonic.
            Indices of monotonic features should be [0:n_monotonic_features].
        use_random_data (bool, optional): Augment the batch with random points. Defaults to False.

    Returns:
        torch.FloatTensor: Value corresponding to regularization penalty
    """

    if use_random_data:
        # a random batch with same shape as data_x is concatenated to data_x prior to mixup.
        data_x = torch.cat([data_x, torch.rand_like(data_x)], 0)

    model.eval()  # Set model to eval mode to use accumulated stats in batch norm layers.

    # Get interpolated data.
    interpolated_data = interpolate_data_batch(data_x, number_of_pairs)

    interpolated_data_monotonic = interpolated_data[:, :n_monotonic_features]
    interpolated_data_non_monotonic = interpolated_data[:, n_monotonic_features:]

    interpolated_data_monotonic.requires_grad = (
        True  # Required since we need to compute grad w.r.t. inputs
    )

    # Predictions are computed to enable gradient computation.
    prediction_from_interpolations = model(
        interpolated_data_monotonic, interpolated_data_non_monotonic,
    )

    grad_wrt_monotonic_input = torch.autograd.grad(
        torch.sum(
            prediction_from_interpolations.max(dim=1)[0]
        ),  # Get gradients w.r.t. max logit.
        interpolated_data_monotonic,  # Only monotonic features are accounted for.
        create_graph=True,
        allow_unused=True,
    )[0]

    grad_wrt_monotonic_input_neg = -grad_wrt_monotonic_input
    grad_wrt_monotonic_input_neg[grad_wrt_monotonic_input_neg < 0.0] = 0.0

    model.train()  # Set model back to train mode.
    return torch.max(grad_wrt_monotonic_input_neg ** 2)
