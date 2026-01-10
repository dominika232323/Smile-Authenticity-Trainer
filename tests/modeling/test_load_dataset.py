import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil
from modeling.load_dataset import load_dataset

class TestLoadDataset:
    @pytest.fixture
    def temp_dir(self):
        temp_dir_path = tempfile.mkdtemp()
        yield Path(temp_dir_path)
        shutil.rmtree(temp_dir_path)

    @pytest.fixture
    def csv_file(self, temp_dir):
        df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6],
            'non_feature1': [7, 8, 9],
            'non_feature2': [10, 11, 12]
        })
        file_path = temp_dir / "test_dataset.csv"
        df.to_csv(file_path, index=False)
        return file_path

    def test_load_dataset_success(self, csv_file):
        non_feature_columns = ['non_feature1', 'non_feature2']
        df = load_dataset(csv_file, non_feature_columns)
        
        assert isinstance(df, pd.DataFrame)
        assert df.shape == (3, 2)
        assert list(df.columns) == ['feature1', 'feature2']
        assert 'non_feature1' not in df.columns
        assert 'non_feature2' not in df.columns

    def test_load_dataset_no_columns_to_drop(self, csv_file):
        non_feature_columns = []
        df = load_dataset(csv_file, non_feature_columns)
        
        assert df.shape == (3, 4)
        assert list(df.columns) == ['feature1', 'feature2', 'non_feature1', 'non_feature2']

    def test_load_dataset_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            load_dataset(Path("non_existent_file.csv"), [])

    def test_load_dataset_missing_column_to_drop(self, csv_file):
        # pandas.DataFrame.drop raises KeyError if column is missing by default
        non_feature_columns = ['missing_column']
        with pytest.raises(KeyError):
            load_dataset(csv_file, non_feature_columns)
