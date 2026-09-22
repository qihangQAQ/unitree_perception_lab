"""Atomic, manifest-backed trajectory shard storage."""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import torch

from .schema import SCHEMA_VERSION, EpisodeData, Split


def _atomic_json_dump(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class EpisodeShardWriter:
    """Write only complete episodes and update the manifest atomically."""

    def __init__(
        self,
        root: str | Path,
        split: str,
        metadata: dict[str, Any],
        *,
        max_frames_per_shard: int = 20_000,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"Invalid split {split!r}.")
        self.root = Path(root)
        self.split = split
        self.split_dir = self.root / split
        self.split_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self.max_frames_per_shard = max_frames_per_shard
        self._episodes: list[dict[str, torch.Tensor]] = []
        self._frames = 0
        self._manifest = self._load_or_create_manifest(metadata)
        self._next_shard = 1 + max(
            (-1, *(int(Path(item["path"]).stem.split("_")[-1]) for item in self._manifest["shards"]))
        )

    def _load_or_create_manifest(self, metadata: dict[str, Any]) -> dict[str, Any]:
        # Normalize tuples and scalar-like values exactly as they will appear
        # after a JSON round trip, so later processes can append safely.
        metadata = json.loads(json.dumps(metadata, sort_keys=True))
        self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            with self.manifest_path.open(encoding="utf-8") as stream:
                manifest = json.load(stream)
            if manifest.get("schema_version") != SCHEMA_VERSION:
                raise ValueError("Cannot append to a dataset with a different schema version.")
            if manifest.get("metadata") != metadata:
                raise ValueError("Cannot append shards whose dataset metadata differs from the manifest.")
            return manifest
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata,
            "shards": [],
        }
        _atomic_json_dump(self.manifest_path, manifest)
        return manifest

    @property
    def next_episode_id(self) -> int:
        return sum(int(item["episodes"]) for item in self._manifest["shards"]) + len(self._episodes)

    def append(self, episode: EpisodeData) -> None:
        episode.validate()
        expected_split = int(Split.from_name(self.split))
        if int(episode.usd_region_split[0]) != expected_split:
            raise ValueError(
                f"Episode spatial split is {int(episode.usd_region_split[0])}, but writer split is {self.split!r}."
            )
        if self._episodes and self._frames + episode.num_frames > self.max_frames_per_shard:
            self.flush()
        self._episodes.append(episode.to_dict())
        self._frames += episode.num_frames

    def flush(self) -> Path | None:
        if not self._episodes:
            return None
        name = f"shard_{self._next_shard:05d}.pt"
        destination = self.split_dir / name
        temporary = self.split_dir / f".{name}.{uuid.uuid4().hex}.tmp"
        payload = {"schema_version": SCHEMA_VERSION, "split": self.split, "episodes": self._episodes}
        torch.save(payload, temporary)
        os.replace(temporary, destination)
        entry = {
            "path": destination.relative_to(self.root).as_posix(),
            "split": self.split,
            "episodes": len(self._episodes),
            "frames": self._frames,
        }
        self._manifest["shards"].append(entry)
        _atomic_json_dump(self.manifest_path, self._manifest)
        self._next_shard += 1
        self._episodes = []
        self._frames = 0
        return destination

    def close(self) -> None:
        self.flush()

    def __enter__(self) -> "EpisodeShardWriter":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def read_manifest(root_or_manifest: str | Path) -> tuple[Path, dict[str, Any]]:
    path = Path(root_or_manifest)
    manifest_path = path if path.name == "manifest.json" else path / "manifest.json"
    with manifest_path.open(encoding="utf-8") as stream:
        manifest = json.load(stream)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version {manifest.get('schema_version')}; expected {SCHEMA_VERSION}.")
    return manifest_path.parent, manifest


def iter_episodes(root_or_manifest: str | Path, split: str) -> Iterator[EpisodeData]:
    root, manifest = read_manifest(root_or_manifest)
    for entry in manifest["shards"]:
        if entry["split"] != split:
            continue
        payload = torch.load(root / entry["path"], map_location="cpu", weights_only=False)
        for data in payload["episodes"]:
            yield EpisodeData.from_dict(data)
