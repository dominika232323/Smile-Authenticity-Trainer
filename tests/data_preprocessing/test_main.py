from pathlib import Path

import numpy as np
import pandas as pd


class TestGetVideosToProcess:
    def _make_mp4_files(self, base: Path, names: list[str]) -> list[Path]:
        base.mkdir(parents=True, exist_ok=True)
        paths = []

        for n in names:
            p = base / f"{n}.mp4"
            p.touch()
            paths.append(p)

        (base / "ignore.txt").write_text("x")

        return paths

    def test_returns_all_when_no_checkpoint(self, tmp_path, monkeypatch):
        from data_preprocessing import main as main_mod

        vids = self._make_mp4_files(tmp_path, ["a", "b", "c"])

        monkeypatch.setattr(main_mod, "UvA_NEMO_SMILE_VIDEOS_DIR", tmp_path)
        monkeypatch.setattr(main_mod, "ensure_checkpoint_file_exists", lambda: False)

        result = main_mod.get_videos_to_process()

        assert isinstance(result, list)
        assert {Path(p) for p in result} == set(vids)

    def test_filters_processed_videos_when_checkpoint_exists(self, tmp_path, monkeypatch):
        from data_preprocessing import main as main_mod

        vids = self._make_mp4_files(tmp_path, ["v1", "v2", "v3"])  # three videos in dir
        monkeypatch.setattr(main_mod, "UvA_NEMO_SMILE_VIDEOS_DIR", tmp_path)

        monkeypatch.setattr(main_mod, "ensure_checkpoint_file_exists", lambda: True)

        processed = pd.DataFrame(
            {
                "file_path": [str(vids[1]), str(tmp_path.parent / "other" / "x.mp4")],
                "done": [1, 1],
            }
        )

        monkeypatch.setattr(pd, "read_csv", lambda _path: processed)
        result = main_mod.get_videos_to_process()
        expected = {vids[0], vids[2]}

        assert {Path(p) for p in result} == expected

    def test_returns_empty_when_no_videos(self, tmp_path, monkeypatch):
        from data_preprocessing import main as main_mod

        monkeypatch.setattr(main_mod, "UvA_NEMO_SMILE_VIDEOS_DIR", tmp_path)
        monkeypatch.setattr(main_mod, "ensure_checkpoint_file_exists", lambda: False)

        result = main_mod.get_videos_to_process()

        assert result == []

    def test_ignores_processed_paths_from_other_directories(self, tmp_path, monkeypatch):
        from data_preprocessing import main as main_mod

        vids = self._make_mp4_files(tmp_path, ["k1", "k2"])
        monkeypatch.setattr(main_mod, "UvA_NEMO_SMILE_VIDEOS_DIR", tmp_path)
        monkeypatch.setattr(main_mod, "ensure_checkpoint_file_exists", lambda: True)

        other_dir = tmp_path.parent / "somewhere_else"
        other_dir.mkdir(exist_ok=True)
        other_file = other_dir / "k1.mp4"
        other_file.touch()

        processed = pd.DataFrame({"file_path": [str(other_file)], "done": [1]})
        monkeypatch.setattr(pd, "read_csv", lambda _path: processed)

        result = main_mod.get_videos_to_process()

        assert {Path(p) for p in result} == set(vids)


class TestPreprocessFrame:
    def test_with_landmarks_saves_and_calls_expected(self, tmp_path, monkeypatch):
        from data_preprocessing import main as main_mod

        monkeypatch.setattr(main_mod, "ORIGINAL_FRAMES_DIR", tmp_path / "orig")
        monkeypatch.setattr(main_mod, "PREPROCESSED_FRAMES_DIR", tmp_path / "preproc")

        video_name = "videoA"
        frame_number = 7
        frame = (np.ones((2, 2, 3)) * 255).astype("uint8")
        orig_lm = tmp_path / "orig.csv"
        norm_lm = tmp_path / "norm.csv"

        calls = {"save": [], "get": [], "norm": []}

        def fake_save_frame(img, path):
            calls["save"].append((img, Path(path)))

        first_call = {"done": False}

        def fake_get_face_landmarks(img, fn, path):
            calls["get"].append((img, fn, Path(path)))

            if not first_call["done"]:
                first_call["done"] = True

                return True

            return None

        def fake_normalize_face(img, lm_path, fn, eye_rel, desired):
            calls["norm"].append((img, Path(lm_path), fn, eye_rel, desired))
            return np.zeros((1, 1, 3), dtype="uint8")

        monkeypatch.setattr(main_mod, "save_frame", fake_save_frame)
        monkeypatch.setattr(main_mod, "get_face_landmarks", fake_get_face_landmarks)
        monkeypatch.setattr(main_mod, "normalize_face", fake_normalize_face)

        main_mod.preprocess_frame(frame, frame_number, video_name, orig_lm, norm_lm)

        expected_orig_path = tmp_path / "orig" / video_name / f"{frame_number}.jpg"

        assert any(p == expected_orig_path for _, p in calls["save"])
        assert len(calls["norm"]) == 1

        n_img, n_lm_path, n_fn, n_eye_rel, n_desired = calls["norm"][0]

        assert n_img is frame
        assert n_lm_path == orig_lm
        assert n_fn == frame_number
        assert n_eye_rel == main_mod.EYE_RELATIVE_SIZE
        assert n_desired == main_mod.DESIRED_FRAME_SIZE

        expected_norm_path = tmp_path / "preproc" / video_name / f"{frame_number}.jpg"

        assert any(p == expected_norm_path for _, p in calls["save"])  # second save
        assert len(calls["save"]) == 2
        assert len(calls["get"]) == 2

        g_img1, g_fn1, g_path1 = calls["get"][0]

        assert g_img1 is frame
        assert g_fn1 == frame_number
        assert g_path1 == orig_lm

        g_img2, g_fn2, g_path2 = calls["get"][1]

        assert g_fn2 == frame_number
        assert g_path2 == norm_lm
        assert isinstance(g_img2, np.ndarray) and g_img2.shape == (1, 1, 3)

    def test_without_landmarks_only_saves_original(self, tmp_path, monkeypatch):
        from data_preprocessing import main as main_mod

        monkeypatch.setattr(main_mod, "ORIGINAL_FRAMES_DIR", tmp_path / "orig")
        monkeypatch.setattr(main_mod, "PREPROCESSED_FRAMES_DIR", tmp_path / "preproc")

        video_name = "videoB"
        frame_number = 3
        frame = np.zeros((2, 2, 3), dtype="uint8")
        orig_lm = tmp_path / "orig2.csv"
        norm_lm = tmp_path / "norm2.csv"

        calls = {"save": [], "get": [], "norm": []}

        def fake_save_frame(img, path):
            calls["save"].append((img, Path(path)))

        def fake_get_face_landmarks(img, fn, path):
            calls["get"].append((img, fn, Path(path)))
            return False

        def fake_normalize_face(*args, **kwargs):
            calls["norm"].append(args)
            return np.ones((1, 1, 3), dtype="uint8")

        monkeypatch.setattr(main_mod, "save_frame", fake_save_frame)
        monkeypatch.setattr(main_mod, "get_face_landmarks", fake_get_face_landmarks)
        monkeypatch.setattr(main_mod, "normalize_face", fake_normalize_face)

        main_mod.preprocess_frame(frame, frame_number, video_name, orig_lm, norm_lm)

        expected_orig_path = tmp_path / "orig" / video_name / f"{frame_number}.jpg"

        assert calls["save"] and calls["save"][0][1] == expected_orig_path
        assert len(calls["save"]) == 1

        assert len(calls["norm"]) == 0
        assert len(calls["get"]) == 1


class TestPreprocessVideo:
    def test_returns_none_when_cannot_open_video(self, tmp_path, monkeypatch):
        from data_preprocessing import main as main_mod

        created = {}

        class DummyCap:
            def __init__(self, _path):
                self.release_called = False

            def isOpened(self):
                return False

            def get(self, _prop):
                return 0

            def set(self, _prop, _value):
                pass

            def read(self):
                return False, None

            def release(self):
                self.release_called = True

        def make_cap(_):
            c = DummyCap(_)
            created["cap"] = c
            return c

        CV2Stub = type(
            "CV2Stub",
            (),
            {
                "CAP_PROP_FPS": 5,
                "CAP_PROP_FRAME_COUNT": 7,
                "CAP_PROP_POS_FRAMES": 1,
                "VideoCapture": make_cap,
            },
        )

        monkeypatch.setattr(main_mod, "cv2", CV2Stub)

        called = {"count": 0}

        def fake_preprocess_frame(*args, **kwargs):
            called["count"] += 1

        monkeypatch.setattr(main_mod, "preprocess_frame", fake_preprocess_frame)

        res = main_mod.preprocess_video(tmp_path / "v.mp4", tmp_path / "orig.csv", tmp_path / "norm.csv")

        assert res is None
        assert called["count"] == 0
        assert created["cap"].release_called is False

    def test_processes_frames_and_returns_fps(self, tmp_path, monkeypatch):
        from data_preprocessing import main as main_mod

        config = {
            "opened": True,
            "fps": 25.0,
            "total": 5,
            "reads": {
                0: (True, np.array([[0]], dtype="uint8")),
                1: (False, None),
                2: (True, np.array([[2]], dtype="uint8")),
                3: (True, np.array([[3]], dtype="uint8")),
                4: (False, None),
            },
        }

        class DummyCap2:
            def __init__(self, _path):
                self.release_called = False
                self.last_pos = None
                self.set_calls = []

            def isOpened(self):
                return config["opened"]

            def get(self, prop):
                if prop == 5:
                    return config["fps"]
                if prop == 7:
                    return config["total"]
                return 0

            def set(self, prop, value):
                self.last_pos = value
                self.set_calls.append((prop, value))

            def read(self):
                return config["reads"].get(self.last_pos, (False, None))

            def release(self):
                self.release_called = True

        created = {}

        def make_cap2(_):
            c = DummyCap2(_)
            created["cap"] = c
            return c

        CV2Stub2 = type(
            "CV2Stub2",
            (),
            {
                "CAP_PROP_FPS": 5,
                "CAP_PROP_FRAME_COUNT": 7,
                "CAP_PROP_POS_FRAMES": 1,
                "VideoCapture": make_cap2,
            },
        )

        monkeypatch.setattr(main_mod, "cv2", CV2Stub2)

        calls = []

        def fake_preprocess_frame(frame, frame_number, video_name, orig_path, norm_path):
            calls.append((frame_number, video_name, frame, orig_path, norm_path))

        monkeypatch.setattr(main_mod, "preprocess_frame", fake_preprocess_frame)

        video_path = tmp_path / "abc.mp4"
        fps = main_mod.preprocess_video(video_path, tmp_path / "orig.csv", tmp_path / "norm.csv")

        assert fps == config["fps"]

        called_frames = [fn for fn, *_ in calls]
        assert called_frames == [0, 2, 3]

        set_positions = [v for (prop, v) in created["cap"].set_calls if prop == CV2Stub2.CAP_PROP_POS_FRAMES]
        assert set_positions == list(range(config["total"]))

        assert created["cap"].release_called is True


class TestSaveLandmarksInApex:
    def test_saves_landmarks_correctly(self, tmp_path, monkeypatch):
        from data_preprocessing import main as main_mod

        # Mock directories
        preproc_dir = tmp_path / "preproc"
        phases_dir = tmp_path / "phases"
        landmarks_dir = tmp_path / "landmarks"

        preproc_dir.mkdir()
        phases_dir.mkdir()
        landmarks_dir.mkdir()

        monkeypatch.setattr(main_mod, "PREPROCESSED_DATA_DIR", preproc_dir)
        monkeypatch.setattr(main_mod, "PREPROCESSED_SMILE_PHASES_DIR", phases_dir)
        monkeypatch.setattr(main_mod, "PREPROCESSED_FACELANDMARKS_DIR", landmarks_dir)

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
        main_mod.save_landmarks_in_apex()

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
