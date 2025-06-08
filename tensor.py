import numpy as np

class Tensor():
    def __init__(self, data, children=(), op=''):
        self.data: np.ndarray = data
        self.grad = np.zeros_like(data)
        self._prev = set(children)
        self._backward = lambda x : None
        self._op = op

    # need to build topo graph and then go through it and call backwards on each of the tensors
    def backward(self):
        self.grad = np.ones_like(self.data)
        pass

    def __add__(self, operand):
        operand = operand if isinstance(operand, Tensor) else Tensor(operand)
        out = Tensor(self.data + operand.data, (self, operand), '+')

        def _backward():
            self.grad += out.grad
            operand.grad += out.grad

        out._backward = _backward

        return out
    
    def __sub__(self, operand):
        pass

    def __mul__(self, operand):
        pass

    def __truediv__(self, operand):
        pass

    def __pow__(self, operand):
        pass

    def relu(self):
        pass
    
    def __matmul__(self, operand):
        operand = operand if isinstance(operand, Tensor) else Tensor(operand)
        out = Tensor(self.data @ operand.data, (self, operand), '@')

        def _backward():
            self.grad += out.grad @ operand.data.T
            operand.grad += self.data.T @ out.grad

        out._backward = _backward

        return out
    
    def sum(self):
        pass

    def mean(self):
        pass
    
    def __repr__(self):
        return f'tensor: {self.data}, grad: {self.grad}, op:{self.op}'

class Relu():
    def __call__(self, input):
        if isinstance(input, Tensor):
            return input.relu()
        else:
            raise TypeError('Input type: {input.type}, not supported')
        
