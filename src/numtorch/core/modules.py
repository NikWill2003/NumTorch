import numpy as np
from typing import Iterable
from abc import abstractmethod


from .tensor import Tensor, Parameter
from ..config import Cfg, get_rng

DTYPE = Cfg.dtype
RNG = get_rng()
    

class Module():
    
    def __call__(self, input: Tensor) -> Tensor:
        return self.forward(input)
    
    @property
    def modules(self) -> list["Module"]:
        modules: list[Module] = []
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

    def train(self) -> None:
        for param in self.params:
            param.requires_grad = True

        for module in self.modules:
            if isinstance(module, DynamicModule):
                module.mode = 'train'
        
    def eval(self) -> None:
        for param in self.params:
            param.requires_grad = False

        for module in self.modules:
            if isinstance(module, DynamicModule):
                module.mode = 'eval'

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

    def forward(self, input: Tensor):
        x = input
        # x: (B, in), A : (in, out), B: out
        return (x @ self.A) + self.b

class Relu(Module):
    def forward(self, x: Tensor):
        return x.relu()
    
class SoftMax(Module):
    def forward(self, x: Tensor):
        # temporary as max is not an implemented op
        x = x - np.max(x.data, axis=-1, keepdims=True) # for numerical stability 
        x = x.exp()
        norm_c = x.sum(axis=-1, keepdims=True)
        return x / norm_c
    
class DynamicModule(Module):
    def __init__(self) -> None:
        self.mode = 'train'

class DropOut(DynamicModule):
    def __init__(self, p):
        self.mode = 'train'
        self.p = p
    
    def forward(self, x: Tensor):
        scale = 1/(1-self.p)
        scaled_mask = RNG.choice([0,scale], size=x.size, p=[self.p, 1-self.p]).reshape(x.shape)

        return (x * scaled_mask) 