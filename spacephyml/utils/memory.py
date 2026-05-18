import numpy as np

class FIFOBuffer:
    def __init__(self, shape, dtype=np.float32):
        self.shape = shape
        if isinstance(shape, tuple):
            self.length = shape[0]
        else:
            self.length = shape
        self.data = np.empty(shape, dtype=dtype)
        self.index = 0
        self.full = False

    def reset(self):
        """
        Reset the FIFO buffer
        """
        self.index = 0
        self.full = False

    def append(self, value):
        """
        Append a value to the buffer.
        """
        self.data[self.index,...] = value
        self.index = (self.index + 1) % self.length
        if self.index == 0:
            self.full = True

    def get(self):
        if not self.full:
            return self.data[:self.index,...]
        # Return in FIFO order
        return np.concatenate((self.data[self.index:,...], self.data[:self.index,...]))

    def __getitem__(self, key):
        """Support indexing and slicing like a NumPy array."""
        arr = self.get()
        return arr[key]

    def __len__(self):
        return self.length if self.full else self.index

    def __repr__(self):
        return f"RingBuffer({self.get()})"
