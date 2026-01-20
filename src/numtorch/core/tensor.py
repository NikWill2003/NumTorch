from __future__ import annotations
import numpy as np
from typing import Optional

from ..config import Cfg, get_rng

DTYPE = Cfg.dtype
RNG = get_rng()

class Tensor():
    def __init__(self, data, requires_grad=False, children=(), op=''):
        self.data: np.ndarray = np.array(data, dtype=DTYPE)
        self.grad = np.zeros_like(data, dtype=DTYPE)
        self.requires_grad = requires_grad
        self._prev = set(children)
        self._backward = lambda : None
        self._op = op

    @property
    def shape(self) -> tuple[int]:
        return self.data.shape
    
    @property
    def size(self) -> int: 
        return self.data.size
    
    @property
    def ndim(self) -> int:
        return self.data.ndim
    
    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=DTYPE)

    def item(self) -> np.ndarray:
        return self.data
    
    def _unbroadcast(self, grad: np.ndarray) -> np.ndarray:
        dims_to_remove = tuple(i for i in range(len(grad.shape) - len(self.shape))) 
        # remove prepended padding dimensions
        grad = np.sum(grad, axis=dims_to_remove, keepdims=False) 
        dims_to_reduce = tuple(i for i, (d1,d2) in enumerate(zip(grad.shape, self.shape)) if d1!=d2)
        # reduce broadcasted dimensions
        return np.sum(grad, axis=dims_to_reduce, keepdims=True)

    # need to build topo graph and then go through it and call backwards on each of the tensors
    def backward(self) -> None:
        self.grad = np.ones_like(self.data)
        topo = []
        visited = set()

        # do DFS on un-visited nodes, add node to topo-when all its children have been visited
        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    build_topo(child)
                topo.append(node)
        build_topo(self)

        for node in reversed(topo):
            node._backward()
            node._prev = set(())
            node._backward = lambda : None

    def __getitem__(self, indexes):
        out = Tensor(self.data[indexes], self.requires_grad, (self,), 'slice')
        
        def _backward():
            if self.requires_grad:
                self.grad[indexes] += out.grad
        out._backward = _backward
        return out
            
    def __add__(self, rhs) -> Tensor:
        rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs)
        out = Tensor(self.data + rhs.data, self.requires_grad or rhs.requires_grad, (self, rhs), '+')

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad)
            if rhs.requires_grad:
                rhs.grad += rhs._unbroadcast(out.grad)
        out._backward = _backward
        return out
    
    def __neg__(self) -> Tensor:
        out = Tensor(-self.data, self.requires_grad, (self,), 'neg')

        def _backward():
            if self.requires_grad:
                self.grad += -out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, rhs) -> Tensor:
        return self + (-rhs)

    def __mul__(self, rhs) -> Tensor:
        rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs)
        out = Tensor(self.data*rhs.data, self.requires_grad or rhs.requires_grad, (self, rhs), f'*')

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad * rhs.data)
            if rhs.requires_grad:
                rhs.grad += rhs._unbroadcast(out.grad * self.data)
        out._backward = _backward
        return out
        
    def __truediv__(self, rhs) -> Tensor:
        return self * (rhs**-1)
    
    # TODO need to sort behaviour here
    def __pow__(self, rhs) -> Tensor: 
        rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs)
        lhs_is_neg = self.data < 0
        rhs_is_frac = ~np.isclose(rhs.data % 1, 0)
        if np.any(lhs_is_neg & rhs_is_frac):
            raise ValueError('cannot raise negative value to a decimal power')
        
        out = Tensor(self.data**rhs.data, self.requires_grad or rhs.requires_grad, (self,), f'**')

        def _backward():
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad * ((rhs.data)*(self.data**(rhs.data-1))))
            if rhs.requires_grad:
                rhs.grad += rhs._unbroadcast(out.grad * (self.data ** rhs.data) * np.log(self.data))
        out._backward = _backward
        return out
    
    '''data shape: (da, ..., d2, d1, n, k) rhs shape: (ob, ..., o2, o1, k, m)
       inputs are broadcast so that they have the same shape by expanding along
       dimensions if possible, out shape: (tc, ..., t2, t1, n, m), where ti = max(di, oi)
       if di or oi does not exist it is treated as 1, and c = max d, a
       if self is 1d shape is prepended with a 1, for rhs it would be appended'''
    def __matmul__(self, rhs) -> Tensor:
        rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs)
        out = Tensor(self.data @ rhs.data, self.requires_grad or rhs.requires_grad, (self, rhs), '@')

        def _backward():
            A, B, = self.data, rhs.data
            g = out.grad
            # broadcast 1d arrays to be 2d 
            A2 = A.reshape(1, -1) if len(A.shape) == 1 else A
            B2 = B.reshape(-1, 1) if len(B.shape) == 1 else B
            # extend g to have reduced dims
            g = np.expand_dims(g, -1) if len(B.shape) == 1 else g
            g = np.expand_dims(g, -2) if len(A.shape) == 1 else g
            # transpose last 2 dimensions, as matmul treats tensors as batched matricies
            if self.requires_grad:
                self.grad += self._unbroadcast(g @ B2.swapaxes(-2, -1))
            if rhs.requires_grad:
                rhs.grad += rhs._unbroadcast(A2.swapaxes(-2, -1) @ g)
        out._backward = _backward
        return out

    def relu(self) -> Tensor:
        out = Tensor((self.data > 0) * self.data, self.requires_grad, (self,), 'Relu')

        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out
    
    def log(self) -> Tensor:
        if np.any(self.data < 0):
            raise ValueError('cannot log negative values')
        out = Tensor(np.log(self.data), self.requires_grad, (self,), 'log')

        def _backward():
            if self.requires_grad:
                self.grad += (1 / self.data) * out.grad
        out._backward = _backward
        return out
    
    def exp(self) -> Tensor:
        out = Tensor(np.exp(self.data), self.requires_grad, (self,), 'exp')

        def _backward():
            if self.requires_grad:
                self.grad += np.exp(self.data) * out.grad
        out._backward = _backward
        return out
    
    def sum(self, axis=None, keepdims=False) -> Tensor:
        out = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), self.requires_grad, (self,), 'sum')

        def _backward():
            if self.requires_grad:
                g = np.expand_dims(out.grad, axis) if (axis is not None and not keepdims) else out.grad
                self.grad += g
        out._backward = _backward
        return out

    def mean(self, axis=None) -> Tensor:
        out = Tensor(np.mean(self.data, axis=axis), self.requires_grad, (self,), 'mean')

        def _backward():
            if self.requires_grad:
                N =  self.size // out.size 
                g = np.expand_dims(out.grad, axis) if axis is not None else out.grad
                self.grad += g / N
        out._backward = _backward
        return out
    
    def clamp(self, a_min=None, a_max=None) -> Tensor:
        out = Tensor(np.clip(self.data, a_min=a_min, a_max=a_max), self.requires_grad, (self,), 'clamp')

        def _backward():
            if self.requires_grad:
                mask = (self.data > a_min) if a_min is not None else np.ones_like(self.data)
                mask = mask & (self.data < a_max) if a_max is not None else mask
                self.grad += out.grad * mask
        out._backward = _backward
        return out

    def __radd__(self, lhs) -> Tensor:
        return self + lhs
    
    def __rsub__(self, lhs) -> Tensor:
        return self - lhs
    
    def __rmul__(self, lhs) -> Tensor:
        return self * lhs
    
    def __rtruediv__(self, lhs) -> Tensor:
        return Tensor(lhs) / self
    
    def __rpow__(self, lhs) -> Tensor:
        return Tensor(lhs) ** self
    
    def __rmatmul__(self, lhs) -> Tensor:
        return Tensor(lhs) @ self
    
    @classmethod
    def random(cls, shape: tuple, bounds = (0,1), requires_grad=False) -> Tensor:
        lower, upper = bounds
        data = RNG.random(shape)*(upper-lower) + lower
        data.astype(DTYPE)
        return cls(data, requires_grad=requires_grad)
    
    def __repr__(self) -> str:
        return f'tensor shape: {self.shape}, op:{self._op}'        

class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(data, requires_grad=True)
    
    @classmethod
    def kaiming(cls, fan_in, shape):
        std = np.sqrt(2/fan_in)
        weights = (RNG.standard_normal(shape, dtype='float32')*std).astype(dtype=DTYPE)
        return cls(weights)
    
    @classmethod
    def zeros(cls, shape):
        return cls(np.zeros(shape, dtype=DTYPE))
    
    def __repr__(self) -> str:
        return f'parameter shape: {self.shape}, size: {self.size}' 