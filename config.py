
class Config:
    """Set default configuration vars."""

    DEVICE = 'cpu'     # 'cuda' or 'cpu'

    NUM_CLASSES = 100   # Number of total classes of CIFAR dataset
    NUM_GROUPS = 10     # Number of total groups to split the dataset

    BATCH_SIZE = int(256 / 4)   # Higher batch sizes allows for larger learning rates. An empirical heuristic
                                # suggests that, when changing the batch size, learning rate should change by the
                                # same factor to have comparable results

    LR = 1e-3            # The initial Learning Rate
    MOMENTUM = 0.9       # Hyperparameter for SGD, keep this at 0.9 when using SGD
    WEIGHT_DECAY = 5e-5  # Regularization, you can keep this at the default

    NUM_EPOCHS = int(1)    # Total number of training epochs (iterations over dataset)
    STEP_SIZE = int(2)    # How many epochs before decreasing learning rate (if using a step-down policy)
    MILESTONES = [49, 63]  # Epochs in which learning rate will be reduced
    GAMMA = 0.1             # Multiplicative factor for learning rate step-down

    LOG_FREQUENCY = 10
