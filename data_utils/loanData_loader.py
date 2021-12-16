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

import random
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


random.seed(10)  # Fixing seed for fixed data splits across experiments

INCREASING_MONOTONIC_FEATURES = (
    "emp_length",
    "annual_income",
    "account_never_delinq_percent",
)

DECREASING_MONOTONIC_FEATURES = (
    "debt_to_income",
    "current_accounts_delinq",
    "num_historical_failed_to_pay",
    "num_collections_last_12m",
    "total_collection_amount_ever",
    "num_accounts_120d_past_due",
    "num_accounts_30d_past_due",
    "public_record_bankrupt",
)

OTHER_FEATURES = (
    "homeownership",
    "verified_income",
    "months_since_last_delinq",
    "earliest_credit_line",
    "inquiries_last_12m",
    "total_credit_lines",
    "open_credit_lines",
    "total_credit_limit",
    "total_credit_utilized",
    "current_installment_accounts",
    "accounts_opened_24m",
    "months_since_last_credit_inquiry",
    "num_satisfactory_accounts",
    "num_active_debit_accounts",
    "total_debit_limit",
    "num_total_cc_accounts",
    "num_open_cc_accounts",
    "num_cc_carrying_balance",
    "num_mort_accounts",
    "tax_liens",
)

TARGET_ID = "loan_amount"

APPLICATION_TYPE_COLUM = "application_type"
APPLICATION_TYPE_FILTER_VALUE = "joint"
JOINT_TO_INDIVIDUAL_COLUMNS = ("annual_income", "debt_to_income")


def _features_normalization(data):
    """Normalizes features.

    Args:
        data (np.array): Batch of train data with shape [TRAIN_DATA_SIZE, N_DIMENSIONS].

    Returns:
        np.array: Normalized data.
    """

    scaler = MinMaxScaler()
    scaler.fit(data)

    return scaler.transform(data)


def _prep_colum_list(data_frame, list_of_columns, invert_direction=False):
    """Pre-process list of columns of a data frame.

    Pre-processing will switch categoricals to numericals, cast entries to float, and
        fill NaN with the average of valid entries.

    Args:
        data_frame (pd.DataFrame): Input data frame to be pre-processed. All elements of
            'list_of_colums' should be in the header of data_frame.
        list_of_columns (tuple): Tuple of header ids one wants to pre-process.
        invert_direction (bool, optional): [description]. Defaults to False.

    Returns:
        np.array: numpy array with processed data from list of columns.
    """

    column_list = []  # List for numpy arrays with column data

    for key in list_of_columns:
        # Cast categoricals to numerical values
        if data_frame[key].dtype == "object":
            data_frame[key] = pd.factorize(data_frame[key])[0]

        # Cast all to float
        data_frame[key] = data_frame[key].astype(np.float64)

        # Filling in nans with average of valid entries
        data_frame[key] = data_frame[key].fillna(data_frame[key].mean())

        # Get numpy array with column data
        column_data = data_frame[key].to_numpy()
        column_data = column_data[
            :, np.newaxis,
        ]  # Creates new dimension for concatenation

        column_list.append(column_data)

    processed_data = np.concatenate(column_list, 1)

    # Flips the signal so monotonicaly decreasing/increasing features become monotonicaly increasing/decreasing
    if invert_direction:
        processed_data = -processed_data

    return processed_data


def load_data(
    path_to_data="./data/loanData/loans_full_schema.csv", valid_split=0.2,
):
    """Loads Loan Lend data.

    Args:
        path_to_data (str, optional): Path to data. Defaults to './data/loanData/loans_full_schema.csv'.
        valid_split (float, optional): value in [0,1] indicating fraction training sample to be used
            for validation. Defaults to 0.2.

    Returns:
        list: Tensors with training and testing data.
    """

    assert (
        valid_split >= 0.0 and valid_split <= 1.0
    ), "valid_split should be in [0.0, 1.0]"

    data = pd.read_csv(path_to_data)

    # The following copies the values from colums finishing in "_joint" to the corresponding columns
    # without the prefix for the rows where APPLICATION_TYPE_COLUM are APPLICATION_TYPE_FILTER_VALUE.
    # E.g. entries in "annual_income" will receive the values of "annual_income_joint" in the rows where
    # application_type is valued "joint"
    for k in JOINT_TO_INDIVIDUAL_COLUMNS:
        data.loc[
            data[APPLICATION_TYPE_COLUM] == APPLICATION_TYPE_FILTER_VALUE, k
        ] = data.loc[
            data[APPLICATION_TYPE_COLUM] == APPLICATION_TYPE_FILTER_VALUE,
            f"{k}_{APPLICATION_TYPE_FILTER_VALUE}",
        ]

    monotonicaly_increasing_features = _prep_colum_list(
        data, INCREASING_MONOTONIC_FEATURES
    )
    monotonicaly_decreasing_features = _prep_colum_list(
        data, DECREASING_MONOTONIC_FEATURES, invert_direction=True
    )
    remaining_features = _prep_colum_list(data, OTHER_FEATURES)
    y = _prep_colum_list(data, (TARGET_ID,))

    X = np.concatenate(
        (
            monotonicaly_increasing_features,
            monotonicaly_decreasing_features,
            remaining_features,
        ),
        1,
    )

    X, y = _features_normalization(X), _features_normalization(y)

    # Random train/validation splitting

    random_indices = list(range(X.shape[0]))
    random.shuffle(random_indices)

    train_data_size = int(X.shape[0] * (1.0 - valid_split))
    train_indices, validation_test_indices = (
        random_indices[:train_data_size],
        random_indices[train_data_size:],
    )

    X_train = X[train_indices, :]
    y_train = y[train_indices, :]
    X_val_test = X[validation_test_indices, :]
    y_val_test = y[validation_test_indices, :]

    # Splits non-training data into validation and test splits
    val_test_idx = X_val_test.shape[0] // 2

    X_val = X_val_test[:val_test_idx, :]
    y_val = y_val_test[:val_test_idx, :]

    X_test = X_val_test[val_test_idx:, :]
    y_test = y_val_test[val_test_idx:, :]

    X_train = torch.tensor(X_train).float()
    X_test = torch.tensor(X_test).float()
    X_val = torch.tensor(X_val).float()
    y_train = torch.tensor(y_train).float()
    y_test = torch.tensor(y_test).float()
    y_val = torch.tensor(y_val).float()

    return X_train, y_train, X_val, y_val, X_test, y_test
