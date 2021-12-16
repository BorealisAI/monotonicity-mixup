# Copyright (c) 2021-present, Royal Bank of Canada.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import random
import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Fixing seed for fixed data across experiments
RANDOM_SEED = 10
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
np.random.seed(seed=RANDOM_SEED)

# List of monotonic and non-monotonic functions to compose function to be approximated
LIST_MONOTONIC_FUNC = [
    lambda x: x / 1e1,
    lambda x: (x + 5.0) / 1e1,
    lambda x: (x - 5.0) / 1e1,
    lambda x: x ** 3 / 1e3,
    lambda x: (x + 5.0) ** 3 / 1e3,
    lambda x: (x - 5.0) ** 3 / 1e3,
]
LIST_NON_MONOTONIC_FUNC = [
    lambda x: x ** 2 / 1e2,
    lambda x: (x + 5.0) ** 2 / 1e2,
    lambda x: (x - 5.0) ** 2 / 1e2,
    lambda x: x ** 4 / 1e4,
    lambda x: (x + 5.0) ** 4 / 1e4,
    lambda x: (x - 5.0) ** 4 / 1e4,
    lambda x: np.sin(x),
    lambda x: np.cos(x),
    lambda x: x * np.sin(x) / 1e1,
    lambda x: x * np.cos(x) / 1e1,
]


def _features_normalization(train_data, test_data):
    """Normalizes train and test features using stats from the train partition.

    Args:
        train_data (torch.FloatTensor): Batch of train data with shape [TRAIN_DATA_SIZE, N_DIMENSIONS].
        test_data (torch.FloatTensor): Batch of test data with shape [TEST_DATA_SIZE, N_DIMENSIONS].

    Returns:
        Tuple: Pair corresponding to normalized train and test data.
    """

    with torch.no_grad():
        scaler = MinMaxScaler()
        scaler.fit(train_data)

        return (
            torch.from_numpy(scaler.transform(train_data)).float(),
            torch.from_numpy(scaler.transform(test_data)).float(),
        )


def load_data(
    path_to_data="./data/synthData/data_{}_{}.pt",
    sample_size=10000,
    n_dimensions=200,
    n_monotonic_dimensions=50,
    fraction_of_dimensions_for_projection=0.3,
    transformation_matrices_shift_coef=0.8,
    valid_split=0.2,
    **kwargs
):
    """Generates synthetic data.

    Covariate shift is further simulated on the test partition.

    Args:
        path_to_data (str, optional): Path to data. Defaults to './data/loanData/loans_full_schema.csv'.
        sample_size (int, optional): Number of examples. Defaults to 10000.
        n_dimensions (int, optional): Number of total timensions. Defaults to 200.
        n_monotonic_dimensions (int, optional): Number of monotonic dimensions. Defaults to 50.
        fraction_of_dimensions_for_projection (float, optional): Value in [0,1] indicating
            the fraction of n_dimensions used to sample the original data. Defaults to 0.2.
        transformation_matrices_shift_coef (float, optional): Value in [0,1] controling
            how much sifted the testa data is relative to train/val. data. Defaults to 0.7.
        valid_split (float, optional): value in [0,1] indicating fraction training sample to be used
            for validation. Defaults to 0.2. Also used to define the size of the test partition.

    Returns:
        list: Tensors with training and testing data.
    """

    data_path = path_to_data.format(n_dimensions, n_monotonic_dimensions)

    # Check if data was created in the past and returns the saved data if so.
    # IMPORTANT: if data creation parameters need to be changed, files must be removed from disk prior to generation.
    if os.path.exists(data_path):
        data = torch.load(data_path)
        return (
            data["x_train"],
            data["y_train"],
            data["x_val"],
            data["y_val"],
            data["x_test"],
            data["y_test"],
        )
    else:  # Data not available. Generate it again.
        # In order to create random data lies in a manifold, we first sample uniformly in low dimension
        # from [-10.0, 10.0] and then expand the dimensions with a random linear transformation.
        train_validation_low_dim_data = (
            20.0
            * torch.rand(
                sample_size, int(fraction_of_dimensions_for_projection * n_dimensions)
            )
            - 10.0
        )
        test_low_dim_data = (
            20.0
            * torch.rand(
                int(valid_split * sample_size),
                int(fraction_of_dimensions_for_projection * n_dimensions),
            )
            - 10.0
        )
        # Matrix to expand dimensions.
        transfromation_matrix = torch.rand(
            int(fraction_of_dimensions_for_projection * n_dimensions), n_dimensions
        )
        # Matrix to expand dimensions. We use a different one to generate test data to simulate covariate shift.
        # We mix the original transformation matrix with the original one to be able to control 'how much' covariate shift
        # is introduced.
        shifted_transfromation_matrix = (
            transformation_matrices_shift_coef
            * torch.rand(
                int(fraction_of_dimensions_for_projection * n_dimensions), n_dimensions
            )
            + (1.0 - fraction_of_dimensions_for_projection) * transfromation_matrix
        )

        with torch.no_grad():
            train_val_data = train_validation_low_dim_data @ transfromation_matrix
            x_test = test_low_dim_data @ shifted_transfromation_matrix

            # Compute train and validation targets
            train_val_targets = []

            for i in range(train_val_data.shape[0]):
                current_target = 0.0
                # Compute monotonic contributions
                for j in range(n_monotonic_dimensions):
                    current_target += LIST_MONOTONIC_FUNC[j % len(LIST_MONOTONIC_FUNC)](
                        train_val_data[i, j]
                    )
                # Compute non-monotonic contributions
                for k in range(n_monotonic_dimensions, n_dimensions):
                    current_target += LIST_NON_MONOTONIC_FUNC[
                        k % len(LIST_NON_MONOTONIC_FUNC)
                    ](train_val_data[i, k])
                train_val_targets.append(current_target)
            train_val_targets = torch.Tensor(train_val_targets).unsqueeze(1)

            # Compute train and validation targets
            test_targets = []

            for i in range(x_test.shape[0]):
                current_target = 0.0
                for j in range(n_monotonic_dimensions):
                    current_target += LIST_MONOTONIC_FUNC[j % len(LIST_MONOTONIC_FUNC)](
                        x_test[i, j]
                    )

                for k in range(n_monotonic_dimensions, n_dimensions):
                    current_target += LIST_NON_MONOTONIC_FUNC[
                        k % len(LIST_NON_MONOTONIC_FUNC)
                    ](x_test[i, k])
                test_targets.append(current_target)
            y_test = torch.Tensor(test_targets).unsqueeze(1)

        # Scale features and target values.
        train_val_data, x_test = _features_normalization(train_val_data, x_test)
        train_val_targets, y_test = _features_normalization(train_val_targets, y_test)

        # Split data into train and validation partitions.
        random_indices = list(range(train_val_data.shape[0]))
        random.shuffle(random_indices)

        train_data_size = int(train_val_data.shape[0] * (1.0 - valid_split))
        train_indices, validation_indices = (
            random_indices[:train_data_size],
            random_indices[train_data_size:],
        )

        x_train = train_val_data[train_indices, :]
        y_train = train_val_targets[train_indices, :]
        x_val = train_val_data[validation_indices, :]
        y_val = train_val_targets[validation_indices, :]

        torch.save(  # saves data for future use
            {
                "x_train": x_train,
                "y_train": y_train,
                "x_val": x_val,
                "y_val": y_val,
                "x_test": x_test,
                "y_test": y_test,
            },
            data_path,
        )

        return x_train, y_train, x_val, y_val, x_test, y_test
