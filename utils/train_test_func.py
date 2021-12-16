# Copyright (c) 2021-present, Royal Bank of Canada.

# All rights reserved.

#

# This source code is licensed under the license found in the

# LICENSE file in the root directory of this source tree.

#

import torch
from torch.nn.modules.loss import MSELoss
from utils.mixup_utils import *
from utils.cmn_utils import *
from utils.networks import *
from utils.common import compute_l1_norm
import advertorch
import advertorch.context


# Setup for evaluation against adversaries
ATTACK = advertorch.attacks.LinfPGDAttack
ADV_CTX = advertorch.context.ctx_noparamgrad_and_eval
EPS = 0.1  # Attack perturbation budget
N_ITERATIONS = 10  # Attack computation budget
# Boundaries of the input space
CLIP_MIN = 0.0
CLIP_MAX = 1.0


def perturb_data(
    model, data, labels, n_monotonic_features, regression=False, device=None
):
    """Adversarial perturbation function.

        Args:
            model (torch.nn.Module): Model to be evaluated.
            data (torch.FloatTensor): Feature matrix.
            labels (torch.LongTensor): Labels.
            n_monotonic_features (int): Number of features with respect to which we expect to be monotonic.
                Indices of monotonic features should be [0:n_monotonic_features].
            regression (bool, optional): Flag to indicate whether we are considering a regression problem
                instead of classification.
            device (torch.device, optional): Device can be passed or the model's device will be used.

        Returns:
            torch.FloatTensor: Perturbed data.
        """

    if device is None:
        device = next(model.parameters()).device

    target_model = wrapper(
        model.eval(), n_monotonic_features
    )  # Creates an attackable model wrapper
    data = data.to(device)
    labels = labels.to(device)

    attack_criterion = torch.nn.MSELoss if regression else torch.nn.BCEWithLogitsLoss

    # Instantiates adversary
    adversary = ATTACK(
        target_model,
        loss_fn=attack_criterion(reduction="sum"),
        eps=EPS,
        nb_iter=N_ITERATIONS,
        eps_iter=EPS / float(N_ITERATIONS),
        rand_init=True,
        clip_min=CLIP_MIN,
        clip_max=CLIP_MAX,
        targeted=False,
    )

    # Perturbs data.
    with ADV_CTX(target_model):
        adv_data = adversary.perturb(data, labels)

    return adv_data


def train(
    model,
    optimizer,
    criterion,
    data_loader,
    l1_coef,
    cmn_lambda,
    mixup_lambda,
    n_monotonic_features,
    regularization_budget,
    reg_mode,
    epoch,
    adversarial_training=False,
):
    """Trains model for one epoch.

    Args:
        model (torch.nn.Module): Model to be evaluated.
        optimizer (torch.optim.Optimizer): Update rule used to update 'model'.
        criterion (e.g. torch.nn.BCEWithLogitsLoss): Loss function.
        data_loader (torch.utils.data.DataLoader): Data loader yielding batches corresponding 
            to tuples (data_batch, labels_batch).
        l1_coef (float): Coefficient for L1 regularization penalty.
        cmn_lambda (float): CMN regularization coefficient.
        mixup_lambda (float): Mixup regularization coefficient.
        n_monotonic_features (int): Number of monotonic features.
        regularization_budget (int): Sample size use to compute regularization penalties.
        reg_mode (str): Indicates the type of monotonicity enforcing penalty.
            Takes values from ["none", "mixup", "cmn", "both"].
        epoch (int): Epoch index. Used for logging only.
        adversarial_training (bool, optional): Enables adversarial training. Defaults to False.
    """

    model.train()
    device = next(model.parameters()).device

    loss_total = 0.0
    reg_loss_mixup = 0.0
    reg_loss_cmn = 0.0
    cmn_grad_penalty_total = 0.0
    mixup_grad_penalty_total = 0.0
    l1_norm_total = 0.0
    batch_idx = 0
    number_of_features = next(iter(data_loader))[0].shape[
        -1
    ]  # Assumes feature matrices with shape [BATCH_SIZE, NUMBER_OF_DIMENSIONS]

    for x, y in iter(data_loader):
        batch_idx += 1
        x, y = x.to(device), y.to(device)

        # If adversarial_training is set, an adversarially perturbed version of the mini-batch
        # is concatenated to the original one
        if adversarial_training:
            X_adv = perturb_data(
                model,
                x,
                y,
                n_monotonic_features,
                regression=isinstance(criterion, nn.MSELoss),
                device=device,
            )
            x = torch.cat((x, X_adv), 0)
            y = torch.cat((y, y), 0)

        optimizer.zero_grad()
        out = model(x[:, :n_monotonic_features], x[:, n_monotonic_features:])
        loss = criterion(out, y)

        if "mixup" in reg_mode:
            reg_loss_mixup = compute_mixup_regularizer(
                model,
                x,
                regularization_budget
                // 2  # If both regularizers are applied, the sample size budget is split between the two methods.
                if "cmn" in reg_mode
                else regularization_budget,
                n_monotonic_features,
                use_random_data=reg_mode == "mixup_random",
            )

        if "cmn" in reg_mode or reg_mode == "train":
            # This case covers the reg modes cmn, cmn_train, cmn_adv, and both.

            if reg_mode == "train":
                split_regularization_budget = x.size(
                    0
                )  # The budget matches the number of training points so no random points will be used.
            elif "mixup" in reg_mode:
                split_regularization_budget = (
                    regularization_budget // 2
                )  # If both regularizers are applied, the sample size budget is split between the two methods.
            else:
                split_regularization_budget = regularization_budget

            reg_loss_cmn = compute_cmn_regularizer(
                model,
                number_of_features,
                n_monotonic_features,
                split_regularization_budget,
                data=x
                if reg_mode == "cmn_train"
                or reg_mode == "cmn_adv"
                or reg_mode == "cmn_mixup_train"
                or reg_mode == "train"
                else None,  # Passes actual train data to be used with random samples.
            )

        # Total loss
        full_loss = loss + cmn_lambda * reg_loss_mixup + mixup_lambda * reg_loss_cmn

        # Adds L1 norm penalty if l1_coef>0.
        if l1_coef > 0.0:
            l1_norm = compute_l1_norm(model)
            full_loss += l1_coef * l1_norm
        else:
            l1_norm = 0.0

        # Accumulates loss terms for logging purposes.
        if epoch is not None:
            loss_total += loss.item()
            if reg_loss_mixup != 0.0:
                mixup_grad_penalty_total += reg_loss_mixup.item()
            if reg_loss_cmn != 0.0:
                cmn_grad_penalty_total += reg_loss_cmn.item()
            if l1_norm != 0.0:
                l1_norm_total += l1_norm.item()

        full_loss.backward()
        optimizer.step()

    # Outputs results to stdout if epoch is passed.
    if epoch is not None:
        print(
            f"Epoch: {epoch}, Loss: {loss_total / batch_idx:.4f}",
            f"Mixup Regularizer: {mixup_grad_penalty_total / batch_idx:.4f}",
            f"CMN Regularizer {cmn_grad_penalty_total / batch_idx:.4f}",
            f"L1 Regularizer {l1_norm_total / batch_idx:.4f}",
        )


def test(model, data, labels, n_monotonic_features, log_info=None, regression=False):
    """Testing function.

    Args:
        model (torch.nn.Module): Model to be evaluated.
        data (torch.FloatTensor): Feature matrix.
        labels (torch.LongTensor): Labels.
        n_monotonic_features (int): Number of features with respect to which we expect to be monotonic.
            Indices of monotonic features should be [0:n_monotonic_features].
        log_info (str, optional): Optional string to be appended to logging output. Defaults to None.
        regression (bool, optional): Flag to indicate whether we are considering a regression problem
            instead of classification.

    Returns:
        float: Prediction accuracy for classification tasks or RMSE if regression is True.
    """

    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    labels = labels.to(device)
    out = model(data[:, :n_monotonic_features], data[:, n_monotonic_features:])

    if regression:
        rmse = torch.nn.MSELoss()(out, labels).sqrt().item()

        if log_info is not None:
            print(f"{log_info}: {rmse:.3f}")

        return rmse
    else:
        # This implicitly sets the threshold at 0.0.
        out[out > 0.0] = 1.0
        out[out < 0.0] = 0.0

        accuracy = torch.sum(out == labels).item() / float(labels.numel())

        if log_info is not None:
            print(f"{log_info}: {accuracy:.3f}")

        return 100.0 * accuracy


def adversarial_test(
    model, data, labels, n_monotonic_features, log_info=None, regression=False
):
    """Adversarial robustness testing function.

    Evaluation is performed after perturbing data adversarially.

    Args:
        model (torch.nn.Module): Model to be evaluated.
        data (torch.FloatTensor): Feature matrix.
        labels (torch.LongTensor): Labels.
        n_monotonic_features (int): Number of features with respect to which we expect to be monotonic.
            Indices of monotonic features should be [0:n_monotonic_features].
        log_info (str, optional): Optional string to be appended to logging output. Defaults to None.
        regression (bool, optional): Flag to indicate whether we are considering a regression problem
            instead of classification.

    Returns:
        float: Robust prediction accuracy for classification tasks or RMSE if regression is True.
    """

    model.eval()
    device = next(model.parameters()).device
    data = data.to(device)
    labels = labels.to(device)

    # Creates adversarial perturbations.
    adv_data = perturb_data(
        model=model,
        data=data,
        labels=labels,
        n_monotonic_features=n_monotonic_features,
        regression=regression,
        device=device,
    )

    out = model(adv_data[:, :n_monotonic_features], adv_data[:, n_monotonic_features:])

    if regression:
        rmse = torch.nn.MSELoss()(out, labels).sqrt().item()

        if log_info is not None:
            print(f"{log_info}: {rmse:.3f}")

        return rmse
    else:
        # This implicitly sets the threshold at 0.0.
        out[out > 0.0] = 1.0
        out[out < 0.0] = 0.0

        accuracy = torch.sum(out == labels).item() / float(labels.numel())

        if log_info is not None:
            print(f"{log_info}: {accuracy:.3f}")

        return 100.0 * accuracy
