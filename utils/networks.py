
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
import torch.nn as nn
import torch.nn.functional as F


class wrapper(nn.Module):
    def __init__(self, base_model, n_monotonic_features):
        """Instantiates model wrapping torch's model.

        This class defines a model wrapping a base classifier in order to
        override its original forward() so that a single argument is passed.
        This is useful for computing adversaries 

        Args:
            base_torch_model (torch.nn.Module): Torch model to be wrapped.
            n_monotonic_features (int): Number of monotonic features.
        """
        super(wrapper, self).__init__()

        self.base_model = base_model
        self.n_monotonic_features = n_monotonic_features

    def forward(self, x):
        """Single argument forward. Used to compute adversaries.

        Args:
            x (torch.FloatTensor): Input data.

        Returns:
            torch.FloatTensor: Model outputs.
        """

        x_monotonic = x[:, : self.n_monotonic_features]
        x_non_monotonic = x[:, self.n_monotonic_features :]

        out = self.base_model(x_monotonic, x_non_monotonic)

        return out


def make_linear_layer(in_size, out_size, activation, use_batch_norm):
    """Makes a Linear layer appended to its activation and optional batch norm.

    Args:
        in_size (int): Input feature size.
        out_size (int): Output feature size.
        activation: e.g. torch.nn.ReLU
        use_batch_norm (bool): Whether to use batch normalization.

    Returns:
        Activated and optionally normalized linear layer.
    """
    # The bias is removed if batch normalization is employed.
    layer = nn.Sequential(nn.Linear(in_size, out_size, bias=not use_batch_norm))
    # Pre-activation batch normalization can be optionally used.
    if use_batch_norm:
        layer.add_module("BN", nn.BatchNorm1d(out_size))
    if activation is not None:
        # Activation function is further appended.
        if isinstance(activation, nn.Hardtanh):
            layer.add_module("Act", activation(min_val=0.0, max_val=1.0))
        else:
            layer.add_module("Act", activation())

    return layer


class base_MLP(nn.Module):
    def __init__(
        self, mono_feature, non_mono_feature, use_batch_norm, *args, **kwargs,
    ):
        super(base_MLP, self).__init__()
        self.hidden_layer = make_linear_layer(
            mono_feature + non_mono_feature,
            (mono_feature + non_mono_feature) // 2,
            nn.LeakyReLU,
            use_batch_norm,
        )

        self.output_layer = make_linear_layer(
            (mono_feature + non_mono_feature) // 2, 1, None, False
        )

    def forward(self, mono_feature, non_mono_feature):

        x = torch.cat((mono_feature, non_mono_feature,), dim=1)
        x = self.hidden_layer(x)
        out = self.output_layer(x)

        return out


class cmn_MLP(nn.Module):
    def __init__(
        self,
        mono_feature,
        non_mono_feature,
        sub_num=1,
        mono_hidden_num=5,
        non_mono_hidden_num=5,
        compress_non_mono=False,
        bottleneck=10,
        use_batch_norm=False,
    ):
        super(cmn_MLP, self).__init__()
        self.compress_non_mono = compress_non_mono
        if compress_non_mono:
            self.non_mono_feature_extractor = make_linear_layer(
                non_mono_feature, 10, nn.Hardtanh, use_batch_norm
            )
            self.mono_fc_in = make_linear_layer(
                mono_feature + 10, mono_hidden_num, nn.ReLU, use_batch_norm
            )
        else:
            self.mono_fc_in = make_linear_layer(
                mono_feature + non_mono_feature,
                mono_hidden_num,
                nn.ReLU,
                use_batch_norm,
            )

        self.non_mono_fc_in = make_linear_layer(
            non_mono_feature, non_mono_hidden_num, nn.ReLU, use_batch_norm
        )
        self.mono_submods_out = nn.ModuleList(
            [
                make_linear_layer(
                    mono_hidden_num, bottleneck, nn.Hardtanh, use_batch_norm,
                )
                for i in range(sub_num)
            ]
        )
        self.mono_submods_in = nn.ModuleList(
            [
                make_linear_layer(
                    2 * bottleneck, mono_hidden_num, nn.ReLU, use_batch_norm,
                )
                for i in range(sub_num)
            ]
        )
        self.non_mono_submods_out = nn.ModuleList(
            [
                make_linear_layer(
                    non_mono_hidden_num, bottleneck, nn.Hardtanh, use_batch_norm,
                )
                for i in range(sub_num)
            ]
        )
        self.non_mono_submods_in = nn.ModuleList(
            [
                make_linear_layer(
                    bottleneck, non_mono_hidden_num, nn.ReLU, use_batch_norm,
                )
                for i in range(sub_num)
            ]
        )

        self.mono_fc_last = make_linear_layer(mono_hidden_num, 1, None, False)
        self.non_mono_fc_last = make_linear_layer(non_mono_hidden_num, 1, None, False)

    def forward(self, mono_feature, non_mono_feature):
        y = self.non_mono_fc_in(non_mono_feature)

        if self.compress_non_mono:
            non_mono_feature = self.non_mono_feature_extractor(non_mono_feature)

        x = self.mono_fc_in(torch.cat([mono_feature, non_mono_feature], dim=1))

        for i in range(int(len(self.mono_submods_out))):
            x = self.mono_submods_out[i](x)

            y = self.non_mono_submods_out[i](y)

            x = self.mono_submods_in[i](torch.cat([x, y], dim=1))

            y = self.non_mono_submods_in[i](y)

        x = self.mono_fc_last(x)

        y = self.non_mono_fc_last(y)

        out = x + y

        return out
