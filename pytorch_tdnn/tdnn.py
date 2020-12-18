# This implementation is based on: https://github.com/jonasvdd/TDNN

import torch
from torch.nn.utils import weight_norm

class TDNN(torch.nn.Module):
    def __init__(self,
                input_dim: int,
                output_dim: int,
                context: list,
                bias: bool = True):
        """
        Implementation of TDNN using the dilation argument of the PyTorch Conv1d class
        Due to its fastness the context has gained two constraints:
            * The context must be symmetric
            * The context must have equal spacing between each consecutive element
        The context can either be of size 2 (e.g. {-1,1} or {-3,3}, like for TDNN-F), or an
        of odd length with 0 in the middle and equal spacing on either side.
        For example: the context {-3, -2, 0, +2, +3} is not valid since it doesn't 
        have equal spacing; The context {-6, -3, 0, 3, 6} is both symmetric and has 
        an equal spacing, this is considered valid.
        :param input_dim: The number of input channels
        :param output_dim: The number of channels produced by the temporal convolution
        :param context: The temporal context
        """
        super(TDNN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        context = sorted(context)
        self.check_valid_context(context)

        kernel_size = len(context)
        if len(context) == 1:
            dilation = 1
            padding = 0
        else:
            delta = [context[i] - context[i - 1] for i in range(1, len(context))]
            dilation = delta[0]
            padding = max(context)
            
        self.temporal_conv = weight_norm(
            torch.nn.Conv1d(
                input_dim,
                output_dim,
                kernel_size=kernel_size,
                dilation=dilation,
                padding=padding,
                bias=bias   # will be set to False for semi-orthogonal TDNNF convolutions
        ))

    def forward(self, x):
        """
        :param x: is one batch of data, x.size(): [batch_size, input_dim, in_seq_length]
            sequence length is the dimension of the arbitrary length data
        :return: [batch_size, output_dim, out_seq_length ]
        """
        return self.temporal_conv(x)

    @staticmethod
    def check_valid_context(context: list) -> None:
        """
        Check whether the context is symmetrical and whether the passed
        context can be used for creating a convolution kernel with dil
        :param context: The context of the model, must be of length 2 or odd, with
            equal spacing.
        """
        assert len(context) == 2 or len(context) % 2 != 0, "Context length must be 2 or odd"
        if len(context) == 2:
            assert context[0] + context[1] == 0, "Context must be of type {-1, 1}" 
        else:
            assert context[len(context) // 2] == 0, "The context contain 0 in the center"
            if len(context) > 1:
                delta = [context[i] - context[i - 1] for i in range(1, len(context))]
                assert all(delta[0] == delta[i] for i in range(1, len(delta))), \
                    "Intra context spacing must be equal!"
        
