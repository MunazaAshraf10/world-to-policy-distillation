from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest

from scripts.extract_occworld_sample import extract_window
from src.data.nuscenes import select_window


def test_official_window_count_and_scene_boundaries() -> None:
    frames = [{"token": str(index)} for index in range(15)]
    scenes = {"scene-0001": frames, "scene-0003": frames}
    scene, offset, selected = select_window(scenes, 3, 12)
    assert (scene, offset) == ("scene-0003", 0)
    assert selected == frames[:12]
    assert select_window(scenes, 2, 12)[1] == 2
    with pytest.raises(IndexError):
        select_window(scenes, 6, 12)
    with pytest.raises(IndexError):
        select_window(scenes, -1, 12)


def test_short_scene_is_rejected() -> None:
    with pytest.raises(ValueError, match="shorter"):
        select_window({"scene-0003": []}, 0, 12)


def test_extract_exact_files_and_preserve_existing(tmp_path: Path) -> None:
    archive = tmp_path / "gts.tar.gz"
    token = "a" * 32
    name = f"gts/scene-0003/{token}/labels.npz"
    with tarfile.open(archive, "w:gz") as stream:
        for entry in ("../outside", "gts/unrelated", name):
            member = tarfile.TarInfo(entry)
            member.size = 4
            stream.addfile(member, io.BytesIO(b"data"))
    root = tmp_path / "nuscenes"
    hashes = extract_window(archive, root, "scene-0003", [token])
    assert list(hashes) == [name]
    assert (root / name).read_bytes() == b"data"
    assert not (tmp_path / "outside").exists()
    assert extract_window(archive, root, "scene-0003", [token]) == hashes
    (root / name).write_bytes(b"changed")
    with pytest.raises(FileExistsError):
        extract_window(archive, root, "scene-0003", [token])
    assert (root / name).read_bytes() == b"changed"


def test_extract_rejects_missing_frame(tmp_path: Path) -> None:
    archive = tmp_path / "empty.tar.gz"
    with tarfile.open(archive, "w:gz"):
        pass
    with pytest.raises(FileNotFoundError):
        extract_window(archive, tmp_path / "nuscenes", "scene-0003", ["a" * 32])


def test_extract_rejects_symlink(tmp_path: Path) -> None:
    archive = tmp_path / "linked.tar.gz"
    token = "b" * 32
    with tarfile.open(archive, "w:gz") as stream:
        member = tarfile.TarInfo(f"gts/scene-0003/{token}/labels.npz")
        member.type = tarfile.SYMTYPE
        member.linkname = "/etc/passwd"
        stream.addfile(member)
    with pytest.raises(ValueError, match="regular"):
        extract_window(archive, tmp_path / "nuscenes", "scene-0003", [token])
