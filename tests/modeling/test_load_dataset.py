import joblib
import numpy as np
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from modeling.load_dataset import load_dataset, feature_selection, scale_data


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
