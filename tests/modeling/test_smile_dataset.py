import numpy as np
import pandas as pd
import torch
import pytest
from modeling.smile_dataset import SmileDataset


class TestSmileDataset:
    @pytest.fixture
    def sample_numpy_data(self):
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        y = np.array([0.0, 1.0, 0.0])
        return X, y

    @pytest.fixture
    def sample_pandas_data(self):
        X = pd.DataFrame({"feat1": [1.0, 3.0, 5.0], "feat2": [2.0, 4.0, 6.0]})
        y = np.array([0.0, 1.0, 0.0])
        return X, y

    def test_init_with_numpy(self, sample_numpy_data):
        X, y = sample_numpy_data
        dataset = SmileDataset(X, y)

        assert isinstance(dataset.X, torch.Tensor)
        assert isinstance(dataset.y, torch.Tensor)
        assert dataset.X.dtype == torch.float32
        assert dataset.y.dtype == torch.float32
        assert torch.equal(dataset.X, torch.tensor(X, dtype=torch.float32))
        assert torch.equal(dataset.y, torch.tensor(y, dtype=torch.float32))

    def test_init_with_pandas(self, sample_pandas_data):
        X, y = sample_pandas_data
        dataset = SmileDataset(X, y)

        assert isinstance(dataset.X, torch.Tensor)
        assert isinstance(dataset.y, torch.Tensor)
        assert dataset.X.dtype == torch.float32
        assert dataset.y.dtype == torch.float32
        assert torch.equal(dataset.X, torch.tensor(X.values, dtype=torch.float32))
        assert torch.equal(dataset.y, torch.tensor(y, dtype=torch.float32))

    def test_len(self, sample_numpy_data):
        X, y = sample_numpy_data
        dataset = SmileDataset(X, y)
        assert len(dataset) == len(y)

    def test_getitem(self, sample_numpy_data):
        X, y = sample_numpy_data
        dataset = SmileDataset(X, y)

        X_item, y_item = dataset[1]

        assert torch.equal(X_item, torch.tensor(X[1], dtype=torch.float32))
        assert torch.equal(y_item, torch.tensor(y[1], dtype=torch.float32))
