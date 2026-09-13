from __future__ import annotations

import argparse
import hashlib
import logging
import pickle
import re
import sys
import tarfile
from pathlib import Path
from typing import Any


def extract_window(archive: Path, data_root: Path, scene: str, tokens: list[str]) -> dict[str, str]:
    """Read only named regular files; existing files must match archive bytes."""
    if re.fullmatch(r"scene-\d{4}", scene) is None or any(
        re.fullmatch(r"[0-9a-f]{32}", token) is None for token in tokens
    ):
        raise ValueError("Expected nuScenes scene name and hexadecimal sample tokens")
    wanted = {f"gts/{scene}/{token}/labels.npz" for token in tokens}
    hashes = {}
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            name = member.name[2:] if member.name.startswith("./") else member.name
            if name not in wanted:
                continue
            if not member.isfile():
                raise ValueError(f"Expected regular occupancy file: {name}")
            source = stream.extractfile(member)
            if source is None:
                raise ValueError(f"Cannot read occupancy file: {name}")
            with source:
                content = source.read()
            digest = hashlib.sha256(content).hexdigest()
            destination = data_root / name
            if destination.exists():
                if hashlib.sha256(destination.read_bytes()).hexdigest() != digest:
                    raise FileExistsError(f"Existing occupancy differs from archive: {destination}")
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as output:
                    output.write(content)
            hashes[name] = digest
            wanted.remove(name)
            if not wanted:
                break
    if wanted:
        raise FileNotFoundError(f"Archive lacks required occupancy files: {sorted(wanted)}")
    return hashes


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root))
    from src.data.nuscenes import select_window
    from src.verification.report import sha256, write_report
    from src.world.types import Rollout

    parser = argparse.ArgumentParser(description="Extract one official OccWorld occupancy window")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--infos", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    if args.manifest.exists():
        raise FileExistsError(f"Manifest already exists: {args.manifest}")
    with args.infos.open("rb") as stream:
        metadata: dict[str, Any] = pickle.load(stream)
    scene, offset, frames = select_window(
        metadata["infos"], args.sample_index, Rollout().sequence_length
    )
    hashes = extract_window(
        args.archive, args.data_root, scene, [frame["token"] for frame in frames]
    )
    write_report(
        args.manifest,
        {
            "archive": str(args.archive.resolve()),
            "archive_sha256": sha256(args.archive),
            "infos": str(args.infos.resolve()),
            "infos_sha256": sha256(args.infos),
            "dataset_index": args.sample_index,
            "scene_name": scene,
            "window_start_index": offset,
            "files_sha256": hashes,
        },
    )
    logging.basicConfig(level=logging.INFO)
    logging.getLogger("data").info(
        "Extracted %d frames from %s; manifest: %s", len(hashes), scene, args.manifest
    )


if __name__ == "__main__":
    main()
