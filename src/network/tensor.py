import numpy as np
from typing import Union, Tuple, Self, Iterable

RNG = np.random.default_rng()
DTYPE = 'float64' 

class Tensor():
    def __init__(self, data, children=(), op=''):
        self.data: np.ndarray = np.array(data, dtype=DTYPE)
        self.grad = np.zeros_like(data, dtype=DTYPE)
        self._prev = set(children)
        self._backward = lambda : None
        self._op = op

    @property
    def shape(self) -> Tuple[int]:
        return self.data.shape
    
    @property
    def size(self) -> int: 
        return self.data.size
    
    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self.data, dtype=DTYPE)

    def item(self) -> np.ndarray:
        return self.data
    
    def _unbroadcast(self, grad: np.ndarray) -> Self:
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
            
    def __add__(self, rhs) -> Self:
        rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs)
        out = Tensor(self.data + rhs.data, (self, rhs), '+')

        def _backward():
            self.grad += self._unbroadcast(out.grad)
            rhs.grad += rhs._unbroadcast(out.grad)
        out._backward = _backward
        return out
    
    def __neg__(self) -> Self:
        out = Tensor(-self.data, (self,), 'neg')

        def _backward():
            self.grad += -out.grad
        out._backward = _backward
        return out
    
    def __sub__(self, rhs) -> Self:
        return self + (-rhs)

    def __mul__(self, rhs) -> Self:
        rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs)
        out = Tensor(self.data*rhs.data, (self,), f'*')

        def _backward():
            self.grad += self._unbroadcast(out.grad * rhs.data)
            rhs.grad += rhs._unbroadcast(out.grad * self.data)
        out._backward = _backward
        return out
        
    def __truediv__(self, rhs) -> Self:
        return self * (rhs**-1)
      
    def __pow__(self, rhs) -> Self: 
        rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs)
        lhs_is_neg = self.data < 0
        rhs_is_frac = ~np.isclose(rhs.data % 1, 0)
        if np.any(lhs_is_neg & rhs_is_frac):
            raise ValueError('cannot raise negative value to a decimal power')
        
        out = Tensor(self.data**rhs.data, (self,), f'**')

        def _backward():
            self.grad += out.grad * ((rhs.data)*(self.data**(rhs.data-1)))
            rhs.grad += out.grad * (self.data ** rhs.data) * np.log(rhs.data)
        out._backward = _backward
        return out
    
    '''data shape: (da, ..., d2, d1, n, k) rhs shape: (ob, ..., o2, o1, k, m)
       inputs are broadcast so that they have the same shape by expanding along
       dimensions if possible, out shape: (tc, ..., t2, t1, n, m), where ti = max(di, oi)
       if di or oi does not exist it is treated as 1, and c = max d, a
       if self is 1d shape is prepended with a 1, for rhs it would be appended'''
    def __matmul__(self, rhs) -> Self:
        rhs = rhs if isinstance(rhs, Tensor) else Tensor(rhs)
        out = Tensor(self.data @ rhs.data, (self, rhs), '@')

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
            self.grad += self._unbroadcast(g @ B2.swapaxes(-2, -1))
            rhs.grad += rhs._unbroadcast(A2.swapaxes(-2, -1) @ g)
        out._backward = _backward
        return out

    def relu(self) -> Self:
        out = Tensor((self.data > 0) * self.data, (self,), 'Relu')

        def _backward():
            self.grad += (self.data > 0) * out.grad
        out._backward = _backward
        return out
    
    # need to check inp is non-negative
    def log(self) -> Self:
        if np.any(self.data < 0):
            raise ValueError('cannot log negative values')
        out = Tensor(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad = self.data ** -1
        out.backward = _backward
        return out
    
    def exp(self) -> Self:
        out = Tensor(np.exp(self.data), (self,), 'exp')

        def _backward():
            self.grad = np.exp(self.data)
        out.backward = _backward
        return out
    
    def sum(self, axis=None) -> Self:
        out = Tensor(np.sum(self.data, axis=axis), (self,), 'sum')

        def _backward():
            g = np.expand_dims(out.grad, axis) if axis is not None else out.grad
            self.grad += g
        out._backward = _backward
        return out

    def mean(self, axis=None) -> Self:
        out = Tensor(np.mean(self.data, axis=axis), (self,), 'mean')

        def _backward():
            N =  self.size // out.size 
            g = np.expand_dims(out.grad, axis) if axis is not None else out.grad
            self.grad += g / N
        out._backward = _backward
        return out
    
    def __radd__(self, lhs) -> Self:
        return self + lhs
    
    def __rsub__(self, lhs) -> Self:
        return self + lhs
    
    def __rmul__(self, lhs) -> Self:
        return self * lhs
    
    def __rtruediv__(self, lhs) -> Self:
        try:
            lhs = Tensor(lhs)
        except TypeError:
            return NotImplementedError
        return lhs / self
    
    def __rpow__(self, lhs) -> Self:
        try:
            lhs = Tensor(lhs)
        except TypeError:
            return NotImplementedError
        return lhs ** self
    
    def __rmatmul__(self, lhs) -> Self:
        try:
            lhs = Tensor(lhs)
        except TypeError:
            return NotImplementedError
        return lhs @ self
    
    @classmethod
    def random(cls, shape: tuple, bounds = (0,1)) -> Self:
        lower, upper = bounds
        data = RNG.random(shape, dtype=DTYPE)*(upper-lower) + lower
        return cls(data)
    
    def __repr__(self) -> str:
        return f'tensor shape: {self.shape}, op:{self._op}'        

