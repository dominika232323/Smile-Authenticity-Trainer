import pandas as pd

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
