from tensor import Tensor
import numpy as np
from abc import ABC, abstractmethod
from typing import Union, Tuple, Self, Iterable

DTYPE = 'float64' 
RNG = np.random.default_rng()

class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data)
    
    @classmethod
    def kaiming(cls, fan_in, shape):
        std = np.sqrt(2/fan_in)
        weights = RNG.standard_normal(shape, dtype=DTYPE)*std
        return cls(weights)
    
    @classmethod
    def zeros(cls, shape):
        return cls(np.zeros(shape, dtype=DTYPE))
    
    def __repr__(self) -> str:
        return f'parameter shape: {self.shape}, size: {self.size}' 
    
class Module(ABC):
    
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)
    
    @property
    def modules(self) -> list[Self]:
        modules: list[Self] = []
        for value in self.__dict__.values():
            if isinstance(value, Module):
                modules.append(value)

            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, Module):
                        modules.append(v)

            elif isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                for v in value:
                    if isinstance(v, Module):
                        modules.append(v)
                    
        return modules
    
    @property
    def params(self) -> list[Parameter]:
        immediate_params = [attr for attr in self.__dict__.values() 
                                    if isinstance(attr, Parameter)]
        modules_params = [param for module in self.modules 
                                    for param in module.params]
        return immediate_params + modules_params
    
    @abstractmethod
    def forward(self, input: Tensor) -> Tensor:
        pass
    
    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

class Sequential(Module):
    def __init__(self, layers):
        self.layers = layers
    
    def forward(self, input: Tensor) -> Tensor:
        x = input
        for layer in self.layers:
            x = layer(x)
        return x
    
class Affine(Module):
    def __init__(self, in_dim, out_dim):
        self.A = Parameter.kaiming(in_dim, (in_dim, out_dim))
        self.b = Parameter.zeros((out_dim))

    def forward(self, x):
        # x: (B, in), A : (in, out), B: out
        return (x @ self.A) + self.b

class Relu(Module):
    def forward(self, x):
        return x.relu()
    
class SoftMaxCrossEntropy():

    def __call__(z: Tensor, y) -> Tensor:
        '''logits z, shape (B, C), true lables y, shape (B, C)'''
        loss = ((z * y).sum(axis=-1) + ((z.exp()).sum(axis=-1)).log()).mean()
        return loss

class SGD():
    def __init__(self, params: list[Parameter], lr: float):
        self.lr = lr
        self.params = params
    
    def step(self) -> None:
        for param in self.params:
            param.data += -self.lr * param.grad