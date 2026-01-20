from math import ceil
import numpy as np

from .core.tensor import Tensor
from .config import get_rng

RNG = get_rng()

class DataLoader():
    def __init__(self, input_data, true_data, batch_size, shuffle=False, rng: np.random.Generator=RNG):
        assert input_data.shape[0] == true_data.shape[0], 'must have the same number of inputs and true outputs'
        
        self.X = input_data
        self.y = true_data
        self.N = batch_size
        self.shuffle = shuffle
        self.rng = rng

    def __iter__(self):
        X, y = self.X, self.y
        if self.shuffle:
            permutation = self.rng.permutation(X.shape[0])
            X = X[permutation]
            y = y[permutation]
        splits = np.arange(self.N, X.shape[0], self.N)
        X = np.split(X, splits, axis=0)
        X = [Tensor(x, requires_grad=False) for x in X]
        y = np.split(y, splits, axis=0)

        return zip(X, y)

    def __len__(self):
        # samples/batch size rounded up
        return ceil(self.X.shape[0]/self.N)

def download_mnist():
    import urllib.request, os

    os.makedirs('datasets', exist_ok=True)
    url = 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz'
    fname, headers = urllib.request.urlretrieve(url, r'datasets/mnist.npz')
    print(f'downloaded mnist to: {fname}')


def get_mnist(normalise: bool = False, flatten: bool = False):

    '''returns (x_train, y_train, x_test, y_test)'''
    try:
        dataset = np.load(r'datasets/mnist.npz')
    except:
        download_mnist()
        dataset = np.load(r'datasets/mnist.npz')
    
    splits: list[np.ndarray] = [dataset['x_train'], dataset['y_train'], dataset['x_test'], dataset['y_test']]
    if normalise:
        splits[0] = splits[0]/255
        splits[2] = splits[2]/255
    if flatten:
        splits[0] = splits[0].reshape(-1, 784)
        splits[2] = splits[2].reshape(-1, 784)

    return splits