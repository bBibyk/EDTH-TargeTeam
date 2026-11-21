from __future__ import annotations

import threading

import numpy as np


class AudioRingBuffer:
    def __init__(self, capacity_samples: int, dtype: np.dtype = np.int16) -> None:
        if capacity_samples <= 0:
            raise ValueError("capacity_samples must be positive")
        self._capacity = int(capacity_samples)
        self._dtype = np.dtype(dtype)
        self._buffer = np.zeros(self._capacity, dtype=self._dtype)
        self._write_index = 0
        self._size = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __len__(self) -> int:
        with self._lock:
            return self._size

    def clear(self) -> None:
        with self._lock:
            self._buffer.fill(0)
            self._write_index = 0
            self._size = 0

    def write(self, samples: np.ndarray) -> None:
        values = np.asarray(samples, dtype=self._dtype).reshape(-1)
        if values.size == 0:
            return

        with self._lock:
            if values.size >= self._capacity:
                self._buffer[:] = values[-self._capacity :]
                self._write_index = 0
                self._size = self._capacity
                return

            end = self._write_index + values.size
            if end <= self._capacity:
                self._buffer[self._write_index : end] = values
            else:
                first_part = self._capacity - self._write_index
                self._buffer[self._write_index :] = values[:first_part]
                self._buffer[: values.size - first_part] = values[first_part:]

            self._write_index = end % self._capacity
            self._size = min(self._capacity, self._size + values.size)

    def snapshot(self, max_samples: int | None = None) -> np.ndarray:
        with self._lock:
            if self._size == 0:
                return np.empty(0, dtype=self._dtype)

            sample_count = self._size
            if max_samples is not None:
                sample_count = min(sample_count, max(0, int(max_samples)))
            if sample_count == 0:
                return np.empty(0, dtype=self._dtype)

            start = (self._write_index - sample_count) % self._capacity
            end = start + sample_count
            if end <= self._capacity:
                return self._buffer[start:end].copy()

            return np.concatenate(
                (self._buffer[start:], self._buffer[: end % self._capacity])
            )
