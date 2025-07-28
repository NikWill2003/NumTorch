# dqn from scratch 

# TODO:
- possibly work on the trainer class,
- add, batch norm
- cli/file logging
- finish off mnist example visulisation,
- implement __getitem__ (look at slicing and indexing in more detail) and boolean ops
- track the gradient norms/ implement gradient tracking
- look at memory size reduction, deleting old tensors?
- test clamp autograd
- add in an argparser
- need to think about how to handle infs, nans, etc, also stability of exps and logs
- make constants/ rngs be in a seperate file
- loss functions, check correctness and test autograd
- add in requires grad functionality (enable grad, context manager etc)
- add in more layers convolutions, (possibly)
- add logging, WandB and also terminal logging
- add in a method for checkpointing/resuming runs
- implement lr schedulers
- add in auto grad visulisation
