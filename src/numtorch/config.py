from dataclasses import dataclass
from typing import Literal, Optional
import numpy as np

Mode = Literal['train', 'eval']

@dataclass
class Cfg:
    dtype: Literal['float16', 'float32', 'float64'] = 'float32'
    dtype2eps = {
        'float16': 1e-4,
        'float32': 1e-7,
        'float64': 1e-15
        }
    
    DEBUG = True
    seed: Optional[int] = None
    
    @property
    def dtype_eps(self) -> float:
        return self.dtype2eps[self.dtype]
    
RNG = np.random.default_rng(seed=Cfg.seed)

def get_rng() -> np.random.Generator:
    return RNG