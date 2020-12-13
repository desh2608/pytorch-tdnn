# pytorch-tdnn

Implementation of Time Delay Neural Network (TDNN) and Factorized TDNN (TDNN-F)
in PyTorch, available as layers which can be used directly.

### Installation

Clone the repository and then do:

```
pip install .
```

or `pip install -e .` for development purpose.

### Usage

There are 2 available TDNN layers: `TDNN` and `FastTDNN`. 

If the contexts are uniformly distanced (e.g. `[-6,-3,0,3,6]`), use `FastTDNN` 
which utilizes the dilation options under PyTorch's `Conv1D` for faster computation.

```
from pytorch_tdnn.tdnn import FastTDNN

tdnn = FastTDNN(
  512, # input dim
  512, # output dim
  [-3,0,3], # context
  full_context=False # if True, use the whole context from -3 to 3
)
```

If the context is non-uniform (e.g. `[-1,0,1,2]` or `[-3,-1,0,1,3]`), use `TDNN`
which uses convolutional masks (and is therefore slower).

```
from pytorch_tdnn.tdnn import TDNN

tdnn = TDNN(
  512, # input dim
  512, # output dim
  [-1,0,1,2], # context
)
```

### Credits

* The TDNN implementation is based on: https://github.com/jonasvdd/TDNN.
* Semi-orthogonal convolutions used in TDNN-F is based on: https://github.com/cvqluu/Factorized-TDNN.

