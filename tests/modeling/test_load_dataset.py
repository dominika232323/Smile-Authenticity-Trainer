import joblib
import numpy as np
import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from torch.utils.data import DataLoader

from modeling.load_dataset import (
    load_dataset,
    feature_selection,
    scale_data,
    split_data,
    get_dataloaders,
    add_prefix,
    load_all_features,
    load_and_apply_feature_selector,
    load_and_apply_scaler,
)


class TestLoadAllFeatures:
    @pytest.fixture
    def temp_dir(self):
        temp_dir_path = tempfile.mkdtemp()
        yield Path(temp_dir_path)
        shutil.rmtree(temp_dir_path)

    @pytest.fixture
    def datasets(self, temp_dir):
        df_lips = pd.DataFrame({"filename": ["1.jpg", "2.jpg"], "feat1": [0.1, 0.2], "label": [0, 1]})
        df_cheeks = pd.DataFrame({"filename": ["1.jpg", "2.jpg"], "feat2": [0.3, 0.4], "label": [0, 1]})
        df_eyes = pd.DataFrame({"filename": ["1.jpg", "2.jpg"], "feat3": [0.5, 0.6], "label": [0, 1]})

        lips_path = temp_dir / "lips.csv"
        cheeks_path = temp_dir / "cheeks.csv"
        eyes_path = temp_dir / "eyes.csv"

        df_lips.to_csv(lips_path, index=False)
        df_cheeks.to_csv(cheeks_path, index=False)
        df_eyes.to_csv(eyes_path, index=False)

        return lips_path, eyes_path, cheeks_path

    def test_load_all_features_success(self, datasets):
        lips_path, eyes_path, cheeks_path = datasets
        result_df = load_all_features(lips_path, eyes_path, cheeks_path)

        assert isinstance(result_df, pd.DataFrame)
        # Expected columns: lips_feat1, cheeks_feat2, eyes_feat3, label
        # filename and original labels should be dropped
        expected_columns = ["lips_feat1", "cheeks_feat2", "eyes_feat3", "label"]
        assert list(result_df.columns) == expected_columns
        assert len(result_df) == 2
        assert result_df["label"].tolist() == [0, 1]

    def test_load_all_features_mismatched_filenames(self, temp_dir):
        df_lips = pd.DataFrame({"filename": ["1.jpg", "2.jpg"], "feat1": [0.1, 0.2], "label": [0, 1]})
        df_cheeks = pd.DataFrame({"filename": ["1.jpg", "3.jpg"], "feat2": [0.3, 0.4], "label": [0, 1]})
        df_eyes = pd.DataFrame({"filename": ["1.jpg", "2.jpg"], "feat3": [0.5, 0.6], "label": [0, 1]})

        lips_path = temp_dir / "lips.csv"
        cheeks_path = temp_dir / "cheeks.csv"
        eyes_path = temp_dir / "eyes.csv"

        df_lips.to_csv(lips_path, index=False)
        df_cheeks.to_csv(cheeks_path, index=False)
        df_eyes.to_csv(eyes_path, index=False)

        # Inner merge should only keep "1.jpg"
        result_df = load_all_features(lips_path, eyes_path, cheeks_path)
        assert len(result_df) == 1
        assert "label" in result_df.columns

    def test_load_all_features_missing_file(self, temp_dir):
        lips_path = temp_dir / "lips.csv"
        eyes_path = temp_dir / "eyes.csv"
        cheeks_path = temp_dir / "cheeks.csv"

        pd.DataFrame({"filename": ["1.jpg"], "f": [1], "label": [0]}).to_csv(lips_path, index=False)
        # eyes_path is missing

        with pytest.raises(FileNotFoundError):
            load_all_features(lips_path, eyes_path, cheeks_path)


class TestAddPrefix:
    def test_add_prefix_success(self):
        df = pd.DataFrame({"feat1": [1, 2], "feat2": [3, 4]})
        prefix = "lips"
        expected_columns = ["lips_feat1", "lips_feat2"]

        result_df = add_prefix(df, prefix)

        assert list(result_df.columns) == expected_columns
        assert (result_df.values == df.values).all()

    def test_add_prefix_special_columns(self):
        df = pd.DataFrame({"filename": ["a.jpg", "b.jpg"], "feat1": [1, 2], "label": [0, 1]})
        prefix = "eyes"
        expected_columns = ["filename", "eyes_feat1", "label"]

        result_df = add_prefix(df, prefix)

        assert list(result_df.columns) == expected_columns

    def test_add_prefix_empty_df(self):
        df = pd.DataFrame(columns=["feat1", "feat2"])
        prefix = "cheeks"
        expected_columns = ["cheeks_feat1", "cheeks_feat2"]

        result_df = add_prefix(df, prefix)

        assert list(result_df.columns) == expected_columns
        assert len(result_df) == 0


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
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        X_train_selected, X_test_selected = feature_selection(X_train, y_train, X_test, how_many_features, temp_dir)

        # Check return type and shape
        assert isinstance(X_train_selected, np.ndarray)
        assert isinstance(X_test_selected, np.ndarray)
        assert X_train_selected.shape == (80, 2)
        assert X_test_selected.shape == (20, 2)

        # Check if selector was saved
        selector_path = temp_dir / "feature_selector.joblib"
        assert selector_path.exists()

        # Check if we can load it back
        selector = joblib.load(selector_path)
        assert hasattr(selector, "get_support")

    def test_feature_selection_all_features(self, temp_dir, sample_data):
        X, y = sample_data
        how_many_features = 4
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)

        X_train_selected, X_test_selected = feature_selection(X_train, y_train, X_test, how_many_features, temp_dir)

        assert X_train_selected.shape == (80, 4)
        assert X_test_selected.shape == (20, 4)
        assert (X_train_selected == X_train.values).all()
        assert (X_test_selected == X_test.values).all()

    def test_feature_selection_invalid_dir(self, sample_data):
        X, y = sample_data
        how_many_features = 2
        X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2)
        invalid_dir = Path("/non_existent_directory_12345")

        # Should raise FileNotFoundError when trying to save joblib
        with pytest.raises(FileNotFoundError):
            feature_selection(X_train, y_train, X_test, how_many_features, invalid_dir)


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
        X = sample_data
        X_train, X_test = X.iloc[:80], X.iloc[80:]
        X_train_scaled, X_test_scaled = scale_data(X_train, X_test, temp_dir)

        # Check return type and shape
        assert isinstance(X_train_scaled, np.ndarray)
        assert isinstance(X_test_scaled, np.ndarray)
        assert X_train_scaled.shape == X_train.shape
        assert X_test_scaled.shape == X_test.shape

        # Check if data is scaled (StandardScaler: mean ~0, std ~1)
        assert np.allclose(X_train_scaled.mean(axis=0), 0, atol=1e-7)
        assert np.allclose(X_train_scaled.std(axis=0), 1, atol=1e-7)

        # Check if scaler was saved
        scaler_path = temp_dir / "scaler.joblib"
        assert scaler_path.exists()

        # Check if we can load it back and it's a scaler
        scaler = joblib.load(scaler_path)
        assert hasattr(scaler, "transform")

    def test_scale_data_consistency(self, temp_dir, sample_data):
        X = sample_data
        X_train, X_test = X.iloc[:80], X.iloc[80:]
        X_train_scaled_1, X_test_scaled_1 = scale_data(X_train, X_test, temp_dir)
        X_train_scaled_2, X_test_scaled_2 = scale_data(X_train, X_test, temp_dir)

        assert np.array_equal(X_train_scaled_1, X_train_scaled_2)
        assert np.array_equal(X_test_scaled_1, X_test_scaled_2)

    def test_scale_data_invalid_dir(self, sample_data):
        X = sample_data
        X_train, X_test = X.iloc[:80], X.iloc[80:]
        invalid_dir = Path("/non_existent_directory_98765")

        with pytest.raises(FileNotFoundError):
            scale_data(X_train, X_test, invalid_dir)


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

    def test_get_dataloaders_no_val_data(self, sample_data):
        X_train, _, y_train, _ = sample_data
        batch_size = 4

        train_loader, val_loader = get_dataloaders(X_train, None, y_train, None, batch_size=batch_size)

        assert isinstance(train_loader, DataLoader)
        assert val_loader is None
        assert len(train_loader.dataset) == len(X_train)


class TestLoadAndApplyFeatureSelector:
    @pytest.fixture
    def temp_dir(self):
        temp_dir_path = tempfile.mkdtemp()
        yield Path(temp_dir_path)
        shutil.rmtree(temp_dir_path)

    @pytest.fixture
    def sample_data(self):
        X = pd.DataFrame(
            {
                "feat1": [1, 2, 3],
                "feat2": [4, 5, 6],
                "feat3": [7, 8, 9],
            }
        )
        return X

    def test_load_and_apply_feature_selector_success(self, temp_dir, sample_data):
        from sklearn.feature_selection import SelectKBest, f_classif

        X = sample_data
        y = np.array([0, 1, 0])

        # Create and fit a selector
        selector = SelectKBest(score_func=f_classif, k=2)
        selector.fit(X, y)

        selector_path = temp_dir / "selector.joblib"
        joblib.dump(selector, selector_path)

        # Apply the function
        X_selected = load_and_apply_feature_selector(X, selector_path)

        # Verify
        assert isinstance(X_selected, pd.DataFrame)
        assert X_selected.shape == (3, 2)
        assert list(X_selected.index) == list(X.index)

        expected_features = X.columns[selector.get_support()]
        assert list(X_selected.columns) == list(expected_features)

        # Verify data consistency
        np.testing.assert_array_equal(X_selected.values, selector.transform(X))

    def test_load_and_apply_feature_selector_file_not_found(self):
        X = pd.DataFrame({"a": [1]})
        invalid_path = Path("/non_existent_path/selector.joblib")

        with pytest.raises(FileNotFoundError):
            load_and_apply_feature_selector(X, invalid_path)


class TestLoadAndApplyScaler:
    @pytest.fixture
    def temp_dir(self):
        temp_dir_path = tempfile.mkdtemp()
        yield Path(temp_dir_path)
        shutil.rmtree(temp_dir_path)

    @pytest.fixture
    def sample_data(self):
        X = pd.DataFrame(
            {
                "feat1": [1.0, 2.0, 3.0],
                "feat2": [4.0, 5.0, 6.0],
            }
        )
        return X

    def test_load_and_apply_scaler_success(self, temp_dir, sample_data):
        from sklearn.preprocessing import StandardScaler

        X = sample_data

        # Create and fit a scaler
        scaler = StandardScaler()
        scaler.fit(X)

        scaler_path = temp_dir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)

        # Apply the function
        X_scaled = load_and_apply_scaler(X, scaler_path)

        # Verify
        assert isinstance(X_scaled, pd.DataFrame)
        assert X_scaled.shape == X.shape
        assert list(X_scaled.index) == list(X.index)
        assert list(X_scaled.columns) == list(X.columns)

        # Verify data consistency
        np.testing.assert_array_almost_equal(X_scaled.values, scaler.transform(X))

    def test_load_and_apply_scaler_file_not_found(self):
        X = pd.DataFrame({"a": [1.0]})
        invalid_path = Path("/non_existent_path/scaler.joblib")

        with pytest.raises(FileNotFoundError):
            load_and_apply_scaler(X, invalid_path)
