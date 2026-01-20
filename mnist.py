from src.numtorch.core.modules import Sequential, DropOut, Affine, Relu
from src.numtorch.core.optim import Adam, SoftMaxCrossEntropy
from src.numtorch.utils import Trainer
from src.numtorch.datasets import DataLoader, get_mnist

wandb_config = {'entity':'niks_priv',
                'project': 'torch from scratch testing',
                'name': 'dropout_tests',
                'config': {'optimiser':'adam', 'lr':0.05},
                'group': 'mnist tests',}

X_train, y_train, X_test, y_test = get_mnist(normalise=True, flatten=True)

p=0.3
nn = Sequential([
    DropOut(p), 
    Affine(784, 200), 
    DropOut(p), 
    Relu(), 
    Affine(200, 100), 
    Relu(), 
    Affine(100, 50), 
    Relu(), 
    Affine(50, 10)
    ])

trainer = Trainer(
    nn,
    Adam(nn.params),
    SoftMaxCrossEntropy(),
    DataLoader(X_train, y_train, 256, shuffle=True),
    DataLoader(X_test, y_test, 256, shuffle=False),
    DataLoader(X_test, y_test, 256, shuffle=False),
    wandb_config=wandb_config
)

trainer.train(20)