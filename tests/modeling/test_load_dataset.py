import joblib
import numpy as np
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from torch.utils.data import DataLoader

from modeling.load_dataset import load_dataset, feature_selection, scale_data, split_data, get_dataloaders


class TestLoadDataset:
    @pytest.fixture
    def temp_dir(self):
        temp_dir_path = tempfile.mkdtemp()
        yield Path(temp_dir_path)
        shutil.rmtree(temp_dir_path)

    @pytest.fixture
    def csv_file(self, temp_dir):
        df = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "non_feature1": [7, 8, 9], "non_feature2": [10, 11, 12]}
        )
        file_path = temp_dir / "test_dataset.csv"
        df.to_csv(file_path, index=False)
        return file_path

    def test_load_dataset_success(self, csv_file):
        non_feature_columns = ["non_feature1", "non_feature2"]
        df = load_dataset(csv_file, non_feature_columns)

        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ["feature1", "feature2"]
        assert "non_feature1" not in df.columns
        assert "non_feature2" not in df.columns

    def test_load_dataset_no_columns_to_drop(self, csv_file):
        non_feature_columns = []
        df = load_dataset(csv_file, non_feature_columns)

        assert df.shape == (3, 4)
        assert list(df.columns) == ["feature1", "feature2", "non_feature1", "non_feature2"]

    def test_load_dataset_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dataset(Path("non_existent_file.csv"), [])

    def test_load_dataset_missing_column_to_drop(self, csv_file):
        # pandas.DataFrame.drop raises KeyError if column is missing by default
        non_feature_columns = ["missing_column"]
        with pytest.raises(KeyError):
            load_dataset(csv_file, non_feature_columns)


class TestFeatureSelection:
    @pytest.fixture
    def temp_dir(self):
        temp_dir_path = tempfile.mkdtemp()
        yield Path(temp_dir_path)
        shutil.rmtree(temp_dir_path)

    @pytest.fixture
    def sample_data(self):
        # Create a simple dataset where some features are clearly better than others
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feat1": np.random.randn(100),
                "feat2": np.random.randn(100),
                "feat3": np.random.randn(100),
                "feat4": np.random.randn(100),
            }
        )
        # y depends on feat1 and feat2
        y = (X["feat1"] + X["feat2"] > 0).astype(int)
        return X, y

    def test_feature_selection_success(self, temp_dir, sample_data):
        X, y = sample_data
        how_many_features = 2

        X_selected = feature_selection(X, y, how_many_features, temp_dir)

        # Check return type and shape
        # Based on sklearn, fit_transform returns a numpy array if input is X
        assert isinstance(X_selected, np.ndarray)
        assert X_selected.shape == (100, 2)

        # Check if selector was saved
        selector_path = temp_dir / "feature_selector.joblib"
        assert selector_path.exists()

        # Check if we can load it back
        selector = joblib.load(selector_path)
        assert hasattr(selector, "get_support")

    def test_feature_selection_all_features(self, temp_dir, sample_data):
        X, y = sample_data
        how_many_features = 4

        X_selected = feature_selection(X, y, how_many_features, temp_dir)

        assert X_selected.shape == (100, 4)
        assert (X_selected == X.values).all()

    def test_feature_selection_invalid_dir(self, sample_data):
        X, y = sample_data
        how_many_features = 2
        invalid_dir = Path("/non_existent_directory_12345")

        # Should raise FileNotFoundError when trying to save joblib
        with pytest.raises(FileNotFoundError):
            feature_selection(X, y, how_many_features, invalid_dir)


class TestScaleData:
    @pytest.fixture
    def temp_dir(self):
        temp_dir_path = tempfile.mkdtemp()
        yield Path(temp_dir_path)
        shutil.rmtree(temp_dir_path)

    @pytest.fixture
    def sample_data(self):
        np.random.seed(42)
        X = pd.DataFrame(
            {
                "feat1": np.random.randn(100) * 10 + 5,
                "feat2": np.random.randn(100) * 0.1 - 2,
                "feat3": np.random.randn(100) + 100,
            }
        )
        return X

    def test_scale_data_success(self, temp_dir, sample_data):
        X_scaled = scale_data(sample_data, temp_dir)

        # Check return type and shape
        assert isinstance(X_scaled, np.ndarray)
        assert X_scaled.shape == sample_data.shape

        # Check if data is scaled (StandardScaler: mean ~0, std ~1)
        assert np.allclose(X_scaled.mean(axis=0), 0, atol=1e-7)
        assert np.allclose(X_scaled.std(axis=0), 1, atol=1e-7)

        # Check if scaler was saved
        scaler_path = temp_dir / "scaler.joblib"
        assert scaler_path.exists()

        # Check if we can load it back and it's a scaler
        scaler = joblib.load(scaler_path)
        assert hasattr(scaler, "transform")

    def test_scale_data_consistency(self, temp_dir, sample_data):
        X_scaled_1 = scale_data(sample_data, temp_dir)
        X_scaled_2 = scale_data(sample_data, temp_dir)

        assert np.array_equal(X_scaled_1, X_scaled_2)

    def test_scale_data_invalid_dir(self, sample_data):
        invalid_dir = Path("/non_existent_directory_98765")

        with pytest.raises(FileNotFoundError):
            scale_data(sample_data, invalid_dir)


class TestSplitData:
    def test_split_data_shapes(self):
        # Prepare dummy data
        n_samples = 100
        n_features = 5
        X = pd.DataFrame(np.random.rand(n_samples, n_features))
        y = np.array([0] * 50 + [1] * 50)  # Balanced classes
        test_size = 0.2

        # Execute
        X_train, X_test, y_train, y_test = split_data(X, y, test_size)

        # Assert shapes
        assert len(X_train) == 80
        assert len(X_test) == 20
        assert len(y_train) == 80
        assert len(y_test) == 20
        assert X_train.shape[1] == n_features

    def test_split_data_stratification(self):
        # Ensure that the class distribution is preserved
        n_samples = 100
        X = pd.DataFrame(np.random.rand(n_samples, 2))
        # 80% class 0, 20% class 1
        y = np.array([0] * 80 + [1] * 20)
        test_size = 0.2

        _, _, _, y_test = split_data(X, y, test_size)

        # In a 20-sample test set, we expect 4 samples of class 1 (20% of 20)
        assert np.sum(y_test == 1) == 4

    def test_split_data_reproducibility(self):
        # The function uses random_state=42, so results should be identical
        X = pd.DataFrame(np.random.rand(10, 2))
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        test_size = 0.5

        X_train1, X_test1, _, _ = split_data(X, y, test_size)
        X_train2, X_test2, _, _ = split_data(X, y, test_size)

        pd.testing.assert_frame_equal(X_train1, X_train2)
        pd.testing.assert_frame_equal(X_test1, X_test2)


class TestGetDataloaders:
    @pytest.fixture
    def sample_data(self):
        X_train = pd.DataFrame(
            {
                "f1": np.random.rand(20),
                "f2": np.random.rand(20),
            }
        )
        y_train = np.random.randint(0, 2, size=20)

        X_val = pd.DataFrame(
            {
                "f1": np.random.rand(10),
                "f2": np.random.rand(10),
            }
        )
        y_val = np.random.randint(0, 2, size=10)

        return X_train, X_val, y_train, y_val

    def test_returns_dataloaders(self, sample_data):
        X_train, X_val, y_train, y_val = sample_data

        train_loader, val_loader = get_dataloaders(X_train, X_val, y_train, y_val, batch_size=4)

        assert isinstance(train_loader, DataLoader)
        assert isinstance(val_loader, DataLoader)

    def test_dataset_lengths(self, sample_data):
        X_train, X_val, y_train, y_val = sample_data

        train_loader, val_loader = get_dataloaders(X_train, X_val, y_train, y_val, batch_size=4)

        assert len(train_loader.dataset) == len(X_train)
        assert len(val_loader.dataset) == len(X_val)

    def test_batch_size(self, sample_data):
        X_train, X_val, y_train, y_val = sample_data
        batch_size = 5

        train_loader, val_loader = get_dataloaders(X_train, X_val, y_train, y_val, batch_size=batch_size)

        train_batch = next(iter(train_loader))
        val_batch = next(iter(val_loader))

        assert len(train_batch[0]) <= batch_size
        assert len(val_batch[0]) <= batch_size

    def test_number_of_batches(self, sample_data):
        X_train, X_val, y_train, y_val = sample_data
        batch_size = 6

        train_loader, val_loader = get_dataloaders(X_train, X_val, y_train, y_val, batch_size=batch_size)

        assert len(train_loader) == int(np.ceil(len(X_train) / batch_size))
        assert len(val_loader) == int(np.ceil(len(X_val) / batch_size))
