import pandas as pd
from pathlib import Path

from data_preprocessing import save_landmarks_in_apex


class TestSaveLandmarksInApex:
    def test_saves_landmarks_correctly(self, tmp_path, monkeypatch):
        import config
        from data_preprocessing import save_landmarks_in_apex as save_mod

        # Mock directories
        preproc_dir = tmp_path / "preproc"
        phases_dir = tmp_path / "phases"
        landmarks_dir = tmp_path / "landmarks"

        preproc_dir.mkdir()
        phases_dir.mkdir()
        landmarks_dir.mkdir()

        monkeypatch.setattr(config, "PREPROCESSED_DATA_DIR", preproc_dir)
        monkeypatch.setattr(config, "PREPROCESSED_SMILE_PHASES_DIR", phases_dir)
        monkeypatch.setattr(config, "PREPROCESSED_FACELANDMARKS_DIR", landmarks_dir)

        monkeypatch.setattr(config, "LIPS_LANDMARKS_IN_APEX_CSV", preproc_dir / "lips_landmarks.csv")
        monkeypatch.setattr(config, "EYES_LANDMARKS_IN_APEX_CSV", preproc_dir / "eyes_landmarks.csv")
        monkeypatch.setattr(config, "CHEEKS_LANDMARKS_IN_APEX_CSV", preproc_dir / "cheeks_landmarks.csv")

        # Also patch them in save_landmarks_in_apex module because they are imported there
        monkeypatch.setattr(save_mod, "PREPROCESSED_DATA_DIR", preproc_dir)
        monkeypatch.setattr(save_mod, "PREPROCESSED_SMILE_PHASES_DIR", phases_dir)
        monkeypatch.setattr(save_mod, "PREPROCESSED_FACELANDMARKS_DIR", landmarks_dir)

        monkeypatch.setattr(save_mod, "LIPS_LANDMARKS_IN_APEX_CSV", preproc_dir / "lips_landmarks.csv")
        monkeypatch.setattr(save_mod, "EYES_LANDMARKS_IN_APEX_CSV", preproc_dir / "eyes_landmarks.csv")
        monkeypatch.setattr(save_mod, "CHEEKS_LANDMARKS_IN_APEX_CSV", preproc_dir / "cheeks_landmarks.csv")

        # Create details.csv
        details_df = pd.DataFrame({"filename": ["vid1.mp4", "vid2.mp4"], "label": ["deliberate", "spontaneous"]})
        details_df.to_csv(preproc_dir / "details.csv", index=False)

        # Mock FaceLandmarks to return small set of indices for easier testing
        from data_preprocessing.face_landmarks import FaceLandmarks

        monkeypatch.setattr(FaceLandmarks, "lips_upper_outer", lambda: [1, 2])
        monkeypatch.setattr(FaceLandmarks, "lips_lower_outer", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "lips_upper_inner", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "lips_lower_inner", lambda: [])

        monkeypatch.setattr(FaceLandmarks, "right_eye_upper_0", lambda: [10])
        monkeypatch.setattr(FaceLandmarks, "right_eye_lower_0", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "right_eye_upper_1", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "right_eye_lower_1", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "right_eye_upper_2", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "right_eye_lower_2", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "right_eye_lower_3", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "left_eye_upper_0", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "left_eye_lower_0", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "left_eye_upper_1", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "left_eye_lower_1", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "left_eye_upper_2", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "left_eye_lower_2", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "left_eye_lower_3", lambda: [])

        monkeypatch.setattr(FaceLandmarks, "left_cheek", lambda: [20])
        monkeypatch.setattr(FaceLandmarks, "right_cheek", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "left_cheek_center", lambda: [])
        monkeypatch.setattr(FaceLandmarks, "right_cheek_center", lambda: [])

        # Create phase and landmark files for vid1
        vid1_phases = pd.DataFrame({"frame_number": [0, 1, 2], "smile_phase": ["onset", "apex", "offset"]})
        vid1_phases.to_csv(phases_dir / "vid1.csv", index=False)

        vid1_landmarks = pd.DataFrame(
            {
                "frame_number": [0, 1, 2],
                "1_x": [10, 11, 12],
                "1_y": [100, 110, 120],
                "2_x": [20, 21, 22],
                "2_y": [200, 210, 220],
                "10_x": [100, 101, 102],
                "10_y": [1000, 1010, 1020],
                "20_x": [200, 201, 202],
                "20_y": [2000, 2010, 2020],
            }
        )
        vid1_landmarks.to_csv(landmarks_dir / "vid1.csv", index=False)

        # Create phase and landmark files for vid2
        vid2_phases = pd.DataFrame({"frame_number": [5, 6], "smile_phase": ["apex", "apex"]})
        vid2_phases.to_csv(phases_dir / "vid2.csv", index=False)

        vid2_landmarks = pd.DataFrame(
            {
                "frame_number": [5, 6],
                "1_x": [51, 61],
                "1_y": [510, 610],
                "2_x": [52, 62],
                "2_y": [520, 620],
                "10_x": [510, 610],
                "10_y": [5100, 6100],
                "20_x": [520, 620],
                "20_y": [5200, 6200],
            }
        )
        vid2_landmarks.to_csv(landmarks_dir / "vid2.csv", index=False)

        # Call function
        save_landmarks_in_apex.save_landmarks_in_apex()

        # Check outputs
        lips_res = pd.read_csv(preproc_dir / "lips_landmarks.csv")
        eyes_res = pd.read_csv(preproc_dir / "eyes_landmarks.csv")
        cheeks_res = pd.read_csv(preproc_dir / "cheeks_landmarks.csv")

        # Verify lips
        # vid1 apex is frame 1. vid2 apex are frames 5, 6.
        # labels: vid1 (deliberate) -> 0, vid2 (spontaneous) -> 1
        assert len(lips_res) == 3
        assert list(lips_res["filename"]) == ["vid1", "vid2", "vid2"]
        assert list(lips_res["frame_number"]) == [1, 5, 6]
        assert list(lips_res["label"]) == [0, 1, 1]
        assert list(lips_res["1_x"]) == [11, 51, 61]
        assert list(lips_res["2_y"]) == [210, 520, 620]

        # Verify eyes
        assert len(eyes_res) == 3
        assert "10_x" in eyes_res.columns
        assert list(eyes_res["10_x"]) == [101, 510, 610]

        # Verify cheeks
        assert len(cheeks_res) == 3
        assert "20_y" in cheeks_res.columns
        assert list(cheeks_res["20_y"]) == [2010, 5200, 6200]


class TestGetLipsIndexes:
    def test_get_lips_indexes_returns_correct_set(self, monkeypatch):
        from data_preprocessing.face_landmarks import FaceLandmarks
        from data_preprocessing.save_landmarks_in_apex import get_lips_indexes

        monkeypatch.setattr(FaceLandmarks, "lips_upper_outer", lambda: [1, 2, 3])
        monkeypatch.setattr(FaceLandmarks, "lips_lower_outer", lambda: [3, 4, 5])
        monkeypatch.setattr(FaceLandmarks, "lips_upper_inner", lambda: [5, 6, 7])
        monkeypatch.setattr(FaceLandmarks, "lips_lower_inner", lambda: [7, 8, 1])

        result = get_lips_indexes()
        assert sorted(result) == [1, 2, 3, 4, 5, 6, 7, 8]


class TestGetEyesIndexes:
    def test_get_eyes_indexes_returns_correct_set(self, monkeypatch):
        from data_preprocessing.face_landmarks import FaceLandmarks
        from data_preprocessing.save_landmarks_in_apex import get_eyes_indexes

        monkeypatch.setattr(FaceLandmarks, "right_eye_upper_0", lambda: [1])
        monkeypatch.setattr(FaceLandmarks, "right_eye_lower_0", lambda: [2])
        monkeypatch.setattr(FaceLandmarks, "right_eye_upper_1", lambda: [3])
        monkeypatch.setattr(FaceLandmarks, "right_eye_lower_1", lambda: [4])
        monkeypatch.setattr(FaceLandmarks, "right_eye_upper_2", lambda: [5])
        monkeypatch.setattr(FaceLandmarks, "right_eye_lower_2", lambda: [6])
        monkeypatch.setattr(FaceLandmarks, "right_eye_lower_3", lambda: [7])
        monkeypatch.setattr(FaceLandmarks, "left_eye_upper_0", lambda: [8])
        monkeypatch.setattr(FaceLandmarks, "left_eye_lower_0", lambda: [9])
        monkeypatch.setattr(FaceLandmarks, "left_eye_upper_1", lambda: [10])
        monkeypatch.setattr(FaceLandmarks, "left_eye_lower_1", lambda: [11])
        monkeypatch.setattr(FaceLandmarks, "left_eye_upper_2", lambda: [12])
        monkeypatch.setattr(FaceLandmarks, "left_eye_lower_2", lambda: [13])
        monkeypatch.setattr(FaceLandmarks, "left_eye_lower_3", lambda: [1])  # Duplicate

        result = get_eyes_indexes()
        assert sorted(result) == list(range(1, 14))


class TestGetCheeksIndexes:
    def test_get_cheeks_indexes_returns_correct_set(self, monkeypatch):
        from data_preprocessing.face_landmarks import FaceLandmarks
        from data_preprocessing.save_landmarks_in_apex import get_cheeks_indexes

        monkeypatch.setattr(FaceLandmarks, "left_cheek", lambda: [100, 101])
        monkeypatch.setattr(FaceLandmarks, "right_cheek", lambda: [102, 103])
        monkeypatch.setattr(FaceLandmarks, "left_cheek_center", lambda: [101, 104])
        monkeypatch.setattr(FaceLandmarks, "right_cheek_center", lambda: [103, 105])

        result = get_cheeks_indexes()
        assert sorted(result) == [100, 101, 102, 103, 104, 105]


class TestGetListOfCoords:
    def test_get_list_of_coords_returns_correct_strings(self):
        from data_preprocessing.save_landmarks_in_apex import get_list_of_coords

        indexes = [1, 5, 10]
        expected = ["1_x", "1_y", "5_x", "5_y", "10_x", "10_y"]
        result = get_list_of_coords(indexes)
        assert result == expected

    def test_get_list_of_coords_with_empty_list(self):
        from data_preprocessing.save_landmarks_in_apex import get_list_of_coords

        assert get_list_of_coords([]) == []


class TestGetLipsEyesCheeksLandmarksForFile:
    def test_returns_correct_dataframes(self, monkeypatch):
        import config
        from data_preprocessing.save_landmarks_in_apex import get_lips_eyes_cheeks_landmarks_for_file
        import data_preprocessing.save_landmarks_in_apex as save_mod

        # Mock directories
        monkeypatch.setattr(config, "PREPROCESSED_SMILE_PHASES_DIR", Path("/fake/phases"))
        monkeypatch.setattr(config, "PREPROCESSED_FACELANDMARKS_DIR", Path("/fake/landmarks"))
        # Also patch in the module where they are imported
        monkeypatch.setattr(save_mod, "PREPROCESSED_SMILE_PHASES_DIR", Path("/fake/phases"))
        monkeypatch.setattr(save_mod, "PREPROCESSED_FACELANDMARKS_DIR", Path("/fake/landmarks"))

        # Mock index-gathering functions
        monkeypatch.setattr(save_mod, "get_lips_indexes", lambda: [1])
        monkeypatch.setattr(save_mod, "get_eyes_indexes", lambda: [2])
        monkeypatch.setattr(save_mod, "get_cheeks_indexes", lambda: [3])

        # Sample data
        filename = "test_vid.mp4"
        label_str = "spontaneous"  # Should map to 1

        def mock_read_csv(path):
            if "phases" in str(path):
                return pd.DataFrame({"frame_number": [10, 11, 12], "smile_phase": ["onset", "apex", "offset"]})
            elif "landmarks" in str(path):
                return pd.DataFrame(
                    {
                        "frame_number": [10, 11, 12],
                        "1_x": [100, 110, 120],
                        "1_y": [101, 111, 121],
                        "2_x": [200, 210, 220],
                        "2_y": [201, 211, 221],
                        "3_x": [300, 310, 320],
                        "3_y": [301, 311, 321],
                    }
                )
            return pd.DataFrame()

        monkeypatch.setattr(pd, "read_csv", mock_read_csv)

        cheeks_df, eyes_df, lips_df = get_lips_eyes_cheeks_landmarks_for_file(filename, label_str)

        # Common checks
        for df in [cheeks_df, eyes_df, lips_df]:
            assert len(df) == 1
            assert df.iloc[0]["smile_phase"] == "apex"
            assert df.iloc[0]["frame_number"] == 11
            assert df.iloc[0]["filename"] == "test_vid"
            assert df.iloc[0]["label"] == 1

        # Specific columns check
        assert list(lips_df.columns) == ["filename", "frame_number", "smile_phase", "1_x", "1_y", "label"]
        assert list(eyes_df.columns) == ["filename", "frame_number", "smile_phase", "2_x", "2_y", "label"]
        assert list(cheeks_df.columns) == ["filename", "frame_number", "smile_phase", "3_x", "3_y", "label"]

        # Values check for lips
        assert lips_df.iloc[0]["1_x"] == 110
        assert lips_df.iloc[0]["1_y"] == 111
