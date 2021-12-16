# Copyright (c) 2021-present, Royal Bank of Canada.
# Copyright (c) 2021-present, https://github.com/gnobitab
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#####################################################################################
# Code is based on the implementation from CMN:
# https://github.com/gnobitab/CertifiedMonotonicNetwork
####################################################################################

import torch
import random


def compute_cmn_regularizer(
    model, feature_dim, n_monotonic_features, sample_size, data=None,
):
    """CMN regularizer as defined in https://github.com/gnobitab/CertifiedMonotonicNetwork/blob/main/compas/train.py#L60

    Args:
        model (torch.nn.module): Maps data to the output space.
        feature_dim (int): Dimensionality of features.
        n_monotonic_features (int): Number of features with respect to which we expect to be monotonic.
            Indices of monotonic features should be [0:n_monotonic_features].
        sample_size (int): Size of random data batch used to compute regularizer.
        data (torch.FloatTensor): Optional batch of data to be used along with random sample
            to compute penalty. If given, expected shape is [BATCH_SIZE, feature_dim].

    Returns:
        torch.FloatTensor: Value corresponding to regularization penalty.
    """

    model.eval()  # Set model to eval mode to use accumulated stats in batch norm layers.

    device = next(model.parameters()).device

    if data is not None:
        data.to(device)
        data_size = data.shape[0]
        if data_size > sample_size:
            # Take random sample respecting budget of points.
            idx = random.sample(list(range(sample_size)))
            input_features = data[idx, :]
        elif data_size == sample_size:  # In this case, only train data is used.
            input_features = data
        else:
            # Complete data with random sample:
            complementary_data = torch.rand(sample_size - data_size, feature_dim).to(
                device
            )
            input_features = torch.cat((data, complementary_data), 0)

    else:
        input_features = torch.rand(sample_size, feature_dim).to(device)

    data_monotonic = input_features[:, :n_monotonic_features]
    data_non_monotonic = input_features[:, n_monotonic_features:]

    data_monotonic.requires_grad = (
        True  # Required since we need to compute grad w.r.t. inputs
    )

    # Predictions are computed to enable gradient computation.
    predictions = model(data_monotonic, data_non_monotonic,)

    # Get gradients with respect to monotonic inputs.
    grad_wrt_monotonic_input = torch.autograd.grad(
        torch.sum(predictions.max(dim=1)[0]),  # Get gradients w.r.t. max logit.
        data_monotonic,  # Only monotonic features are accounted for.
        create_graph=True,
        allow_unused=True,
    )[0]

    grad_wrt_monotonic_input_neg = -grad_wrt_monotonic_input
    grad_wrt_monotonic_input_neg[grad_wrt_monotonic_input_neg < 0.0] = 0.0
    model.train()  # Set model back to train mode.
    return torch.max(grad_wrt_monotonic_input_neg ** 2)
