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


class TestMain:
    def test_main_success(self, tmp_path, monkeypatch):
        from data_preprocessing import main as main_mod

        # 1. Setup mock directories and files
        mock_dirs = {
            "PREPROCESSED_DATA_DIR": tmp_path / "preprocessed_data",
            "ORIGINAL_FRAMES_DIR": tmp_path / "original_frames",
            "ORIGINAL_FACELANDMARKS_DIR": tmp_path / "original_landmarks",
            "PREPROCESSED_FRAMES_DIR": tmp_path / "preprocessed_frames",
            "PREPROCESSED_FACELANDMARKS_DIR": tmp_path / "preprocessed_landmarks",
            "PREPROCESSED_SMILE_PHASES_DIR": tmp_path / "preprocessed_phases",
            "LIP_FEATURES_DIR": tmp_path / "lip_features",
            "EYES_FEATURES_DIR": tmp_path / "eyes_features",
            "CHEEKS_FEATURES_DIR": tmp_path / "cheeks_features",
        }
        for attr, path in mock_dirs.items():
            monkeypatch.setattr(main_mod, attr, path)
            path.mkdir(parents=True, exist_ok=True)

        mock_files = {
            "UvA_NEMO_SMILE_DETAILS": tmp_path / "smile_details.csv",
            "CHECKPOINT_FILE_PATH": tmp_path / "checkpoint.csv",
            "ALL_LIP_FEATURES_CSV": tmp_path / "all_lips.csv",
            "ALL_EYES_FEATURES_CSV": tmp_path / "all_eyes.csv",
            "ALL_CHEEKS_FEATURES_CSV": tmp_path / "all_cheeks.csv",
        }
        for attr, path in mock_files.items():
            monkeypatch.setattr(main_mod, attr, path)

        # 2. Mock external functions
        calls = []

        def tracked_mock(name, return_value=None):
            def wrapper(*args, **kwargs):
                calls.append((name, args, kwargs))
                return return_value

            return wrapper

        monkeypatch.setattr(main_mod, "setup_logging", tracked_mock("setup_logging"))
        monkeypatch.setattr(main_mod, "create_directories", tracked_mock("create_directories"))

        details_df = pd.DataFrame({"filename": ["v1.mp4"], "label": ["deliberate"]})
        monkeypatch.setattr(main_mod, "get_details", tracked_mock("get_details", details_df))

        video_path = tmp_path / "v1.mp4"
        video_path.touch()
        monkeypatch.setattr(main_mod, "get_videos_to_process", tracked_mock("get_videos_to_process", [video_path]))

        monkeypatch.setattr(
            main_mod, "create_facelandmarks_header", tracked_mock("create_facelandmarks_header", ["h1", "h2"])
        )
        monkeypatch.setattr(main_mod, "create_csv_with_header", tracked_mock("create_csv_with_header"))
        monkeypatch.setattr(main_mod, "preprocess_video", tracked_mock("preprocess_video", 25.0))
        monkeypatch.setattr(main_mod, "label_smile_phases", tracked_mock("label_smile_phases"))

        lip_df = pd.DataFrame({"feat": [1]})
        monkeypatch.setattr(main_mod, "extract_lip_features", tracked_mock("extract_lip_features", lip_df))
        eye_df = pd.DataFrame({"feat": [2]})
        monkeypatch.setattr(main_mod, "extract_eye_features", tracked_mock("extract_eye_features", eye_df))
        cheek_df = pd.DataFrame({"feat": [3]})
        monkeypatch.setattr(main_mod, "extract_cheek_features", tracked_mock("extract_cheek_features", cheek_df))

        monkeypatch.setattr(main_mod, "save_dataframe_to_csv", tracked_mock("save_dataframe_to_csv"))
        monkeypatch.setattr(main_mod, "append_row_to_csv", tracked_mock("append_row_to_csv"))

        monkeypatch.setattr(main_mod, "concat_csvs", tracked_mock("concat_csvs", pd.DataFrame({"feat": [1]})))
        monkeypatch.setattr(
            main_mod, "assign_labels", tracked_mock("assign_labels", pd.DataFrame({"feat": [1], "label": [0]}))
        )
        monkeypatch.setattr(main_mod, "save_landmarks_in_apex", tracked_mock("save_landmarks_in_apex"))

        # 3. Run main
        main_mod.main()

        # 4. Verify calls
        call_names = [c[0] for c in calls]
        assert "setup_logging" in call_names
        assert "create_directories" in call_names
        assert "get_details" in call_names
        assert "get_videos_to_process" in call_names
        assert "preprocess_video" in call_names
        assert "label_smile_phases" in call_names
        assert "extract_lip_features" in call_names
        assert "extract_eye_features" in call_names
        assert "extract_cheek_features" in call_names
        assert "save_landmarks_in_apex" in call_names

        # Verify details were saved
        details_path = mock_dirs["PREPROCESSED_DATA_DIR"] / "details.csv"
        assert details_path.exists()

    def test_main_get_details_failure(self, tmp_path, monkeypatch):
        from data_preprocessing import main as main_mod

        # Setup mock directories
        monkeypatch.setattr(main_mod, "PREPROCESSED_DATA_DIR", tmp_path / "preproc")
        monkeypatch.setattr(main_mod, "UvA_NEMO_SMILE_DETAILS", tmp_path / "details.csv")

        def fail_get_details(_path):
            raise Exception("Failed to load")

        monkeypatch.setattr(main_mod, "get_details", fail_get_details)
        monkeypatch.setattr(main_mod, "create_directories", lambda x: None)

        # This should return early without calling get_videos_to_process
        mock_get_vids_called = []
        monkeypatch.setattr(main_mod, "get_videos_to_process", lambda: mock_get_vids_called.append(1))

        main_mod.main()

        assert len(mock_get_vids_called) == 0

    def test_main_skips_missing_video(self, tmp_path, monkeypatch):
        from data_preprocessing import main as main_mod

        # Setup mocks
        monkeypatch.setattr(main_mod, "PREPROCESSED_DATA_DIR", tmp_path / "preproc")
        monkeypatch.setattr(main_mod, "ORIGINAL_FRAMES_DIR", tmp_path / "orig_frames")
        monkeypatch.setattr(main_mod, "PREPROCESSED_FRAMES_DIR", tmp_path / "pre_frames")
        (tmp_path / "preproc").mkdir()

        details_df = pd.DataFrame({"filename": ["v1.mp4"], "label": ["deliberate"]})
        monkeypatch.setattr(main_mod, "get_details", lambda _: details_df)

        video_path = tmp_path / "non_existent.mp4"
        # NOT touching the file
        monkeypatch.setattr(main_mod, "get_videos_to_process", lambda: [video_path])

        preprocess_called = []
        monkeypatch.setattr(main_mod, "preprocess_video", lambda *args: preprocess_called.append(args))
        monkeypatch.setattr(main_mod, "create_directories", lambda *args: None)
        monkeypatch.setattr(main_mod, "create_csv_with_header", lambda *args: None)
        monkeypatch.setattr(main_mod, "save_landmarks_in_apex", lambda: None)
        monkeypatch.setattr(main_mod, "assign_labels", lambda df, _: df)
        monkeypatch.setattr(main_mod, "concat_csvs", lambda _: pd.DataFrame())
        monkeypatch.setattr(main_mod, "save_dataframe_to_csv", lambda *args: None)

        main_mod.main()

        assert len(preprocess_called) == 0

    def test_main_skips_if_preprocess_video_returns_none(self, tmp_path, monkeypatch):
        from data_preprocessing import main as main_mod

        monkeypatch.setattr(main_mod, "PREPROCESSED_DATA_DIR", tmp_path / "preproc")
        monkeypatch.setattr(main_mod, "ORIGINAL_FRAMES_DIR", tmp_path / "orig_frames")
        monkeypatch.setattr(main_mod, "PREPROCESSED_FRAMES_DIR", tmp_path / "pre_frames")
        monkeypatch.setattr(main_mod, "ORIGINAL_FACELANDMARKS_DIR", tmp_path / "orig_landmarks")
        monkeypatch.setattr(main_mod, "PREPROCESSED_FACELANDMARKS_DIR", tmp_path / "pre_landmarks")
        monkeypatch.setattr(main_mod, "LIP_FEATURES_DIR", tmp_path / "lip_features")
        monkeypatch.setattr(main_mod, "EYES_FEATURES_DIR", tmp_path / "eyes_features")
        monkeypatch.setattr(main_mod, "CHEEKS_FEATURES_DIR", tmp_path / "cheeks_features")
        (tmp_path / "preproc").mkdir()

        details_df = pd.DataFrame({"filename": ["v1.mp4"], "label": ["deliberate"]})
        monkeypatch.setattr(main_mod, "get_details", lambda _: details_df)

        video_path = tmp_path / "v1.mp4"
        video_path.touch()
        monkeypatch.setattr(main_mod, "get_videos_to_process", lambda: [video_path])

        # preprocess_video returns None
        monkeypatch.setattr(main_mod, "preprocess_video", lambda *args: None)

        label_smile_called = []
        monkeypatch.setattr(main_mod, "label_smile_phases", lambda *args: label_smile_called.append(args))

        monkeypatch.setattr(main_mod, "create_directories", lambda *args: None)
        monkeypatch.setattr(main_mod, "create_csv_with_header", lambda *args: None)
        monkeypatch.setattr(main_mod, "save_landmarks_in_apex", lambda: None)
        monkeypatch.setattr(main_mod, "assign_labels", lambda df, _: df)
        monkeypatch.setattr(main_mod, "concat_csvs", lambda _: pd.DataFrame())
        monkeypatch.setattr(main_mod, "save_dataframe_to_csv", lambda *args: None)
        monkeypatch.setattr(main_mod, "create_facelandmarks_header", lambda: [])

        main_mod.main()

        assert len(label_smile_called) == 0


