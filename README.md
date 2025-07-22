# dqn from scratch 

# TODO:
- finish off mnist example visulisation,
- fix bug where r ops dont work on non-tensors
- implement __getitem__ (look at slicing and indexing in more detail) and boolean ops
- track the gradient norms/ implement gradient tracking
- look at memory size reduction, deleting old tensors?
- test clamp autograd
- possibly work on the trainer class,
- add in an argparser
- need to think about how to handle infs, nans, etc, also stability of exps and logs
- stability of code when using float 32
- make constants/ rngs be in a seperate file
- loss functions, check correctness and test autograd
- add in requires grad functionality (enable grad, context manager etc)
- add in more layers convolutions, softmax, dropout, batch norm (possibly)
- add logging, WandB and also terminal logging
- add in a method for checkpointing/resuming runs
- implement lr schedulers
- add in auto grad visulisation
