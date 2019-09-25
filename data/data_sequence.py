import keras
import math
import numpy as np

from typing import List, Callable, Dict, Tuple, Optional, Collection


class DataSequence(keras.utils.Sequence):
    x: List
    batch_size: int
    length: int
    shuffle: bool
    keep_remainder: bool
    map_fn: Callable
    batches: List[Tuple[int, int]]
    indices: np.ndarray

    def __init__(self,
                 x: Collection,
                 batch_size: int = 32,
                 shuffle: bool = True,
                 map_fn: Optional[Callable] = None,
                 keep_remainder: bool = True) -> None:
        assert isinstance(batch_size, int) and batch_size > 0
        self.batch_size = batch_size
        self.shuffle = bool(shuffle)
        self.keep_remainder = bool(keep_remainder)
        data_length = len(x)  # Implicitly check that x has length
        self.x = list(x)
        # self.map_fn = (lambda inputs, labels: (inputs, labels)) if map_fn is None else map_fn
        self.map_fn = map_fn
        assert callable(self.map_fn)
        if self.keep_remainder:
            self.length = math.ceil(data_length / batch_size)
        else:
            self.length = math.floor(data_length / batch_size)
        assert self.length > 0  # There are some batches to iterate through

        self.batches = [(i * batch_size, min(data_length, (i + 1) * batch_size)) for i in range(self.length)]
        self.indices = np.arange(data_length)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index: int):
        batch_start, batch_end = self.batches[index]
        batch_indices = self.indices[batch_start:batch_end]
        return self.map_fn([self.x[i] for i in batch_indices])

    def __len__(self):
        return self.length

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
