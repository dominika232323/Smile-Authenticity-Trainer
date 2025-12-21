import pandas as pd

from data_preprocessing.assign_labels import assign_labels


class TestAssignLabels:
    def test_assign_labels_basic(self, tmp_path):
        df_features = pd.DataFrame({"filename": ["a.jpg", "b.jpg", "c.jpg"], "f1": [0.1, 0.2, 0.3]})

        df_labels = pd.DataFrame(
            {"filename": ["a.jpg", "b.jpg", "c.jpg"], "label": ["deliberate", "spontaneous", "spontaneous"]}
        )

        csv_path = tmp_path / "labels.csv"
        df_labels.to_csv(csv_path, index=False)

        result = assign_labels(df_features, csv_path)
        expected = [0, 1, 1]

        assert list(result["label"]) == expected
        assert list(result["f1"]) == [0.1, 0.2, 0.3]

    def test_assign_labels_missing_labels(self, tmp_path):
        df_features = pd.DataFrame({"filename": ["x.png", "y.png"], "f1": [1.0, 2.0]})

        df_labels = pd.DataFrame(
            {
                "filename": ["x.png"],
                "label": ["deliberate"],
            }
        )

        csv_path = tmp_path / "details.csv"
        df_labels.to_csv(csv_path, index=False)

        result = assign_labels(df_features, csv_path)

        assert result.loc[result["filename"] == "x.png", "label"].iloc[0] == 0
        assert pd.isna(result.loc[result["filename"] == "y.png", "label"].iloc[0])

    def test_assign_labels_unexpected_label(self, tmp_path):
        df_features = pd.DataFrame(
            {
                "filename": ["img1.png"],
            }
        )

        df_labels = pd.DataFrame(
            {
                "filename": ["img1.png"],
                "label": ["unknown_label"],
            }
        )
        csv_path = tmp_path / "labels.csv"
        df_labels.to_csv(csv_path, index=False)

        result = assign_labels(df_features, csv_path)

        assert pd.isna(result["label"].iloc[0])
