from pathlib import Path

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
        from ai.data_preprocessing import main as main_mod

        vids = self._make_mp4_files(tmp_path, ["a", "b", "c"])

        monkeypatch.setattr(main_mod, "UvA_NEMO_SMILE_VIDEOS_DIR", tmp_path)
        monkeypatch.setattr(main_mod, "ensure_checkpoint_file_exists", lambda: False)

        result = main_mod.get_videos_to_process()

        assert isinstance(result, list)
        assert {Path(p) for p in result} == set(vids)
        assert all(isinstance(p, Path) for p in result)

    def test_filters_processed_videos_when_checkpoint_exists(self, tmp_path, monkeypatch):
        from ai.data_preprocessing import main as main_mod

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
        from ai.data_preprocessing import main as main_mod

        monkeypatch.setattr(main_mod, "UvA_NEMO_SMILE_VIDEOS_DIR", tmp_path)
        monkeypatch.setattr(main_mod, "ensure_checkpoint_file_exists", lambda: False)

        result = main_mod.get_videos_to_process()

        assert result == []

    def test_ignores_processed_paths_from_other_directories(self, tmp_path, monkeypatch):
        from ai.data_preprocessing import main as main_mod

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
