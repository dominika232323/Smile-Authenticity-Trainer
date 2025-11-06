import pytest
import pandas as pd
from pathlib import Path
import tempfile
import shutil

from ai.data_preprocessing.get_details import get_details


@pytest.fixture
def temp_dir():
    temp_dir_path = tempfile.mkdtemp()
    yield Path(temp_dir_path)
    shutil.rmtree(temp_dir_path)


@pytest.fixture
def create_test_file(temp_dir):
    def _create_file(content: str, filename: str = "test_details.txt") -> Path:
        file_path = temp_dir / filename
        with open(file_path, "w") as f:
            f.write(content)
        return file_path

    return _create_file


@pytest.mark.parametrize(
    "content,expected_rows",
    [
        (
            """# Header line 1
# Header line 2
# Header line 3
# Header line 4
file001.jpg    SUB001    M    25    positive""",
            1,
        ),
        (
            """# Header line 1
# Header line 2
# Header line 3
# Header line 4
file001.jpg    SUB001    M    25    positive
file002.jpg    SUB002    F    30    negative""",
            2,
        ),
        (
            """# Header line 1
# Header line 2
# Header line 3
# Header line 4
file001.jpg    SUB001    M    25    positive
file002.jpg    SUB002    F    30    negative
file003.jpg    SUB003    M    28    positive""",
            3,
        ),
    ],
)
def test_get_details_valid_file(create_test_file, content, expected_rows):
    file_path = create_test_file(content)
    result = get_details(file_path)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == expected_rows
    assert list(result.columns) == ["filename", "subject_code", "gender", "age", "label"]

    assert result.iloc[0]["filename"] == "file001.jpg"
    assert result.iloc[0]["subject_code"] == "SUB001"
    assert result.iloc[0]["gender"] == "M"
    assert result.iloc[0]["age"] == 25
    assert result.iloc[0]["label"] == "positive"


def test_get_details_file_not_found(temp_dir):
    non_existent_path = temp_dir / "non_existent.csv"

    with pytest.raises(FileNotFoundError, match=f"File not found: {non_existent_path}"):
        get_details(non_existent_path)


def test_get_details_empty_file(create_test_file):
    file_path = create_test_file("")
    result = get_details(file_path)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert list(result.columns) == ["filename", "subject_code", "gender", "age", "label"]


def test_get_details_file_with_only_headers(create_test_file):
    content = """# Header line 1
# Header line 2
# Header line 3
# Header line 4"""

    file_path = create_test_file(content)
    result = get_details(file_path)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 0
    assert list(result.columns) == ["filename", "subject_code", "gender", "age", "label"]


def test_get_details_malformed_data(create_test_file):
    content = """# Header line 1
# Header line 2
# Header line 3
# Header line 4
file001.jpg    SUB001    M    25    positive
incomplete_row    SUB002
file003.jpg    SUB003    F    invalid_age    negative"""

    file_path = create_test_file(content)
    result = get_details(file_path)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert list(result.columns) == ["filename", "subject_code", "gender", "age", "label"]


def test_get_details_different_separators(create_test_file):
    content = """# Header line 1
# Header line 2
# Header line 3
# Header line 4
file001.jpg\tSUB001\tM\t25\tpositive
file002.jpg    SUB002    F    30    negative
file003.jpg SUB003 M 28 positive"""

    file_path = create_test_file(content)

    result = get_details(file_path)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 3
    assert result.iloc[0]["filename"] == "file001.jpg"
    assert result.iloc[1]["subject_code"] == "SUB002"


def test_get_details_large_file(create_test_file):
    header = """# Header line 1
# Header line 2
# Header line 3
# Header line 4
"""
    rows = []
    for i in range(1000):
        rows.append(
            f"file{i:03d}.jpg    SUB{i:03d}    {'M' if i % 2 == 0 else 'F'}    {25 + (i % 50)}    {'positive' if i % 3 == 0 else 'negative'}"
        )

    content = header + "\n".join(rows)
    file_path = create_test_file(content)

    result = get_details(file_path)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 1000
    assert list(result.columns) == ["filename", "subject_code", "gender", "age", "label"]
