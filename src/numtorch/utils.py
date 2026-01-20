import wandb
from typing import Any, Optional
from dotenv import load_dotenv
import os
import numpy as np

from .core.tensor import Tensor
from .core.optim import optimiser, Loss_Fn
from .core.modules import Module

from .datasets import DataLoader

class MultiAverageMeter():
    def __init__(self):
        self._means = {}
        self._counts = {}

    def update_one(self, metric: str, mean: float, n: int = 1):  
        if metric in self._means.keys():
            k = self._counts[metric]
            self._means[metric] += n*(mean - self._means[metric])/(k+n)
            self._counts[metric] += n
        else:
            self._means[metric] = mean
            self._counts[metric] = n

    def update_many(self, metric_dict: dict[str, float|list]):
        for metric, val in metric_dict.items():
            if isinstance(val, float):
                mean = val
                n = 1
            elif isinstance(val, (list, tuple)):
                mean = val[0]
                n = val[1]
            self.update_one(metric, mean, n)
    
    def get_metric(self, metric):
        if metric in self._means:
            return self._means[metric]
        raise KeyError(f'{metric} not found')
    
    def dump_metrics(self):
        return self._means

    def reset(self, metric=None):
        if metric is None:
            self._means = {}
            self._counts = {}
        else:
            del self._means[metric]
            del self._counts[metric]

    def get_log_str(self, metrics=None):
        log_str = ''
        metrics = self._means.keys() if metrics is None else metrics
        for metric in metrics:
            if metric not in self._means:
                continue
            log_str += f'{metric} : {self._means[metric]:.4f} '
        return log_str

    def __getitem__(self, key):  
        return self.get_metric(key)
    
    def __contains__(self, key): 
        return key in self._means
    
    def __iter__(self):          
        return iter(self._means)
    
    def items(self):            
        return self._means.items()
    
    def __len__(self):           
        return len(self._means)
    
    def __repr__(self):          
        return f'MultiAverageMeter({self._means})'

class WandBLogger():
    def __init__(self, wandb_config: dict[str, Any] = {}, api_key:Optional[str]=None):
        
        self._api_key = api_key if api_key is not None else self.get_api_key()
        assert self._api_key is not None, 'api key required'

        self._mode = 'train'

        logged_in = wandb.login(key=self._api_key)
        assert logged_in; ConnectionError('failed to connect to weights and biases')
        self._wandb_run = wandb.init(**wandb_config)

        self._current_update = {}

    @staticmethod
    def get_api_key():
        load_dotenv()
        return os.getenv('WANDB_API_KEY')
    
    def stage_metrics(self, meter: MultiAverageMeter, mode: str):

        for name, val in meter.items():
            self._current_update[name + f'/{mode}'] = val

    def on_epoch_end(self, epoch: int): 

        if self._wandb_run is not None:
            self._wandb_run.log(self._current_update, step=epoch)

    def on_train_end(self): 
        if self._wandb_run is not None:
            self._wandb_run.finish()

def accuracy(out: Tensor, y: np.ndarray) -> float:
    y_pred = out.data.argmax(axis=-1)
    accuracy = ((y_pred == y).sum())/y.size
    return accuracy

class Trainer():
    def __init__(
            self, 
            model: Module, 
            optimiser: optimiser, 
            loss_fn: Loss_Fn, 
            train_loader: DataLoader, 
            validation_loader: DataLoader, 
            test_loader: Optional[DataLoader],
            wandb_config: dict[str, Any],
            wandb_api_key: Optional[str]=None):
        
        self.model = model
        self.optimiser = optimiser
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.test_loader = test_loader
        self._epoch = 1
        self._meter = MultiAverageMeter()
        try:
            self._wandb_logger = WandBLogger(wandb_config, wandb_api_key)
        except:
            print('wandb init failed')
            self._wandb_logger = None

        self._mode: str = 'train'

    def train_batch(self, X, y):
        self.optimiser.zero_grad()
        output = self.model(X)
        loss = self.loss_fn(output, y)
        loss.backward()
        self.optimiser.step()

        return (loss, output)

    def train_epoch(self):
        self.model.train()

        epoch_meter = MultiAverageMeter()

        for X, y in self.train_loader:

            loss, out = self.train_batch(X, y)
            acc = accuracy(out, y)
            metrics_dict = {'loss': (loss.item(), X.shape[0]),
                            'acc': (acc, X.shape[0]),} 
            epoch_meter.update_many(metrics_dict) # type: ignore

        return epoch_meter

    def evaluate_batch(self, X, y):
        self.optimiser.zero_grad()
        output = self.model(X)
        loss = self.loss_fn(output, y)

        return (loss, output)

    def evaluate(self):
        self.model.eval()

        epoch_meter = MultiAverageMeter()

        for X, y in self.validation_loader:

            loss, out = self.evaluate_batch(X, y)
            acc = accuracy(out, y)
            metrics_dict = {'loss': (loss.item(), X.shape[0]),
                            'acc': (acc, X.shape[0]),} 
            epoch_meter.update_many(metrics_dict) # type: ignore

        return epoch_meter   

    def test(self):
        if self.test_loader is None: 
            return None
        
        self.model.eval()
        epoch_meter = MultiAverageMeter()

        for X, y in self.test_loader:

            loss, out = self.evaluate_batch(X, y)
            acc = accuracy(out, y)
            metrics_dict = {'loss': (loss.item(), X.shape[0]),
                            'acc': (acc, X.shape[0]),} 
            self._meter.update_many(metrics_dict) #type: ignore
        
        return epoch_meter

    def train(self, epochs: int):
        try:
            for epoch in range(epochs):
                print(f'epoch: {epoch}')
                train_meter = self.train_epoch()
                if self._wandb_logger is not None:
                    self._wandb_logger.stage_metrics(train_meter, 'train')
                print('train: ' + train_meter.get_log_str())


                eval_meter = self.evaluate()
                if self._wandb_logger is not None:
                    self._wandb_logger.stage_metrics(eval_meter, 'eval')
                print('eval: ' + eval_meter.get_log_str())

                if self._wandb_logger is not None:
                    self._wandb_logger.on_epoch_end(epoch)

            if self.test_loader is not None:
                test_meter = self.test()
                if test_meter:
                    print('test: ' + test_meter.get_log_str())
        finally:
            if self._wandb_logger is not None:
                self._wandb_logger.on_train_end()

        