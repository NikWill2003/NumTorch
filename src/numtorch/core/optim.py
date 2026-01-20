import numpy as np
from typing import Iterable
from abc import abstractmethod

from .tensor import Tensor, Parameter
from ..config import Cfg, get_rng

DTYPE = Cfg.dtype
RNG = get_rng()
DTYPE_EPS = Cfg.dtype_eps

class Loss_Fn():
    def __call__(self, *args, **kwargs) -> Tensor:
        raise NotImplementedError("Loss function must implement __call__ method")
    
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'
    
    def __str__(self) -> str:
        return self.__repr__()

class SoftMaxCrossEntropy(Loss_Fn):
    def __call__(self, z: Tensor, y: Tensor | np.ndarray) -> Tensor:
        '''logits z, shape (B, C), true integer lables y, shape (B)'''

        assert z.ndim == 2
        assert (y.ndim == 1 or (y.ndim == 2 and y.shape[-1] == 1))
        assert z.shape[0] == y.shape[0]
        
        d0_idxs = np.arange(z.shape[0])
        if y.ndim == 2:
            d0_idxs = d0_idxs.reshape(-1, 1)
        d1_idxs = y.data if isinstance(y, Tensor) else y
        idxs = (d0_idxs, d1_idxs)

        loss = (-(z[idxs]) + ((z.exp()).sum(axis=-1)).log()).mean()

        return loss

class CrossEntropy(Loss_Fn):
    def __call__(self, q: Tensor, y: Tensor | np.ndarray) -> Tensor:
        '''pred q, shape (B, C), true integer lables y, shape (B)'''
        
        assert q.ndim == 2
        assert (y.ndim == 1 or (y.ndim == 2 and y.shape[-1] == 1))
        
        d0_idxs = np.arange(q.shape[0])
        if y.ndim == 2:
            d0_idxs = d0_idxs.reshape(-1, 1)
        d1_idxs = y.data if isinstance(y, Tensor) else y
        idxs = (d0_idxs, d1_idxs)

        loss = -(q[idxs] + DTYPE_EPS).log().mean()
        
        return loss
    
class MeanSquaredError():
    def __call__(self, q: Tensor, y) -> Tensor:
        '''pred q, shape (B, C), true values y, shape (B, C)'''
        loss = ((q - y) ** 2).sum(axis=-1).mean()
        return loss

class optimiser():
    def __init__(self, params: list[Parameter], lr: float=0.005):
        self.lr = lr
        self.params = params
    
    @abstractmethod
    def step(self) -> None:
        pass

    def zero_grad(self) -> None:
        for param in self.params:
            param.zero_grad()

    def train(self) -> None:
        for param in self.params:
            param.requires_grad = True
    
    def eval(self) -> None:
        for param in self.params:
            param.requires_grad = False
        
    
class SGD(optimiser):
    
    def step(self) -> None:
        for param in self.params:
            if not param.requires_grad:
                continue 
            param.data += -self.lr * param.grad

class Adam(optimiser):
    def __init__(
            self, params: list[Parameter], lr: float=0.005, 
            b1: float = 0.9, b2: float = 0.999, eps: float=1e-8
        ):
        
        super().__init__(params, lr)
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.time_step = 0
        self.m = [np.zeros_like(param.data, dtype=DTYPE) for param in params]
        self.v = [np.zeros_like(param.data, dtype=DTYPE) for param in params]
    
    def step(self) -> None:
        self.time_step += 1
        for i, p in enumerate(self.params):
            if not p.requires_grad:
                continue 

            g = p.grad
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*g
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(g**2)
            m_hat = self.m[i]/(1-self.b1**self.time_step)
            v_hat = self.v[i]/(1-self.b2**self.time_step)

            p.data += -self.lr * m_hat / (v_hat ** 0.5 + self.eps)