# pytorch-tdnn

Implementation of Time Delay Neural Network (TDNN) and Factorized TDNN (TDNN-F)
in PyTorch, available as layers which can be used directly.

### Setup

For using (no development required)

```bash
pip install pytorch-tdnn
```

To install for development, clone the repository, and then run the following from
within the roor directory.

```bash
pip install -e .
``` 

### Usage

#### Using the TDNN layer

```python
from pytorch_tdnn.tdnn import TDNN as TDNNLayer

tdnn = TDNNLayer(
  512, # input dim
  512, # output dim
  [-3,0,3], # context
)

y = tdnn(x)
```

Here, `x` should have the shape `(batch_size, sequence_length, input_dim)`. 

**Note:** The `context` list should follow these constraints:
  * The length of the list should be 2 or an odd number.
  * If the length is 2, it should be of the form `[-1,1]` or `[-3,3]`, but not
  `[-1,3]`, for example.
  * If the length is an odd number, they should be evenly spaced with a 0 in the
  middle. For example, `[-3,0,3]` is allowed, but `[-3,-1,0,1,3]` is not.

#### Using the TDNNF layer

```python
from pytorch_tdnn.tdnnf import TDNNF as TDNNFLayer

tdnnf = TDNNFLayer(
  512, # input dim
  512, # output dim
  256, # bottleneck dim
  1, # time stride
)

y = tdnnf(x, training=True)
```

The argument `training` is used to perform the semi-orthogonality step only during
the model training. If this call is made from within a `forward()` function of an
`nn.Module` class, `training` can be set to `self.training`.

**Note:** Time stride should be greater than or equal to 0. For example, if
the time stride is 1, a context of `[-1,1]` is used for each stage of splicing.

### Credits

* The TDNN implementation is based on: https://github.com/jonasvdd/TDNN.
* Semi-orthogonal convolutions used in TDNN-F are based on: https://github.com/cvqluu/Factorized-TDNN.

This repository aims to wrap up these implementations in easy-installable PyPi
packages, which can be used directly in PyTorch based neural network training.

### Issues

If you find any bugs in the code, please raise an Issue, or email me at
`r.desh26@gmail.com`.
