# This implemetation is based on: https://github.com/jonasvdd/TDNN

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class TDNN(nn.Module):
  def __init__(self,
               input_dim: int,
               output_dim: int,
               context: list):
    """
    Implementation of a TDNN layer which uses weight masking to create non symmetric convolutions
    :param input_dim: The hidden dimension of previous layer
    :param output_dim: The number of output dimensions
    :param context: The temporal context
    """
    super(TDNN, self).__init__()

    # create the convolution mask
    self.conv_mask = self._create_conv_mask(context)

    # TDNN convolution
    self.temporal_conv = weight_norm(nn.Conv1d(input_dim, output_dim,
                           kernel_size=self.conv_mask.size()[0]))

    # expand the mask and register a hook to zero gradients during backprop
    self.conv_mask = self.conv_mask.expand_as(self.temporal_conv.weight)
    self.temporal_conv.weight.register_backward_hook(lambda grad: grad * self.conv_mask)

  def forward(self, x):
    """
    :param x: is one batch of data, x.size(): [batch_size, input_dim, sequence_length]
      sequence length is the dimension of the arbitrary length data
    :return: [batch_size, output_dim, sequence_length - kernel_size + 1]
    """
    return self.temporal_conv(x)

  @staticmethod
  def _create_conv_mask(context: list) -> torch.Tensor:
    """
    :param context: The temporal context
      TODO some more exlanation about the convolution
    :return: The convolutional mask
    """
    context = sorted(context)
    min_pos, max_pos = context[0], context[-1]
    mask = torch.zeros(size=(max_pos - min_pos + 1,), dtype=torch.int8)
    context = list(map(lambda x:  x-min_pos, context))
    mask[context] = 1
    return mask


class FastTDNN(nn.Module):
  def __init__(self,
               input_dim: int,
               output_dim: int,
               context: list,
               full_context: bool = True):
    """
    Implementation of a 'Fast' TDNN layer by exploiting the dilation argument of the PyTorch Conv1d class
    Due to its fastness the context has gained two constraints:
      * The context must be symmetric
      * The context must have equal spacing between each consecutive element
    For example: the non-full and symmetric context {-3, -2, 0, +2, +3} is not valid since it doesn't have
    equal spacing; The non-full context {-6, -3, 0, 3, 6} is both symmetric and has an equal spacing, this is
    considered valid.
    :param input_dim: The number of input channels
    :param output_dim: The number of channels produced by the temporal convolution
    :param context: The temporal context
    :param full_context: Indicates whether a full context needs to be used
    """
    super(FastTDNN, self).__init__()
    self.full_context = full_context
    self.input_dim = input_dim
    self.output_dim = output_dim

    context = sorted(context)
    self.check_valid_context(context, full_context)

    if full_context:
      kernel_size = context[-1] - context[0] + 1 if len(context) > 1 else 1
      self.temporal_conv = weight_norm(nn.Conv1d(input_dim, output_dim, kernel_size))
    else:
      # use dilation
      if len(context) > 1:
        delta = [context[i] - context[i - 1] for i in range(1, len(context))]
        dil = delta[0]
        pad = max(context)
      else:
        dil = 1
        pad = 0
      self.temporal_conv = weight_norm(
        nn.Conv1d(input_dim, output_dim, kernel_size=len(context), dilation=dil, padding=pad))

  def forward(self, x):
    """
    :param x: is one batch of data, x.size(): [batch_size, input_dim, sequence_length]
      sequence length is the dimension of the arbitrary length data
    :return: [batch_size, output_dim, len(valid_steps)]
    """
    return self.temporal_conv(x)

  @staticmethod
  def check_valid_context(context: list, full_context: bool) -> None:
    """
    Check whether the context is symmetrical and whether the passed
    context can be used for creating a convolution kernel with dil
    :param full_context: indicates whether the full context (dilation=1) will be used
    :param context: The context of the model, must be symmetric if no full context and have an equal spacing.
    """
    if full_context:
      assert len(context) <= 2, "If the full context is given one must only define the smallest and largest"
      if len(context) == 2:
        assert context[0] + context[-1] == 0, "The context must be symmetric"
    else:
      assert len(context) % 2 != 0, "The context size must be odd"
      assert context[len(context) // 2] == 0, "The context contain 0 in the center"
      if len(context) > 1:
        delta = [context[i] - context[i - 1] for i in range(1, len(context))]
        assert all(delta[0] == delta[i] for i in range(1, len(delta))), "Intra context spacing must be equal!"
        
