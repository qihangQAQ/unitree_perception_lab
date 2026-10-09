"""Atomically published, versioned memory-mapped *samples*, not raw episodes."""

from __future__ import annotations

import errno
import hashlib
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import torch


CACHE_VERSION = 1  # Bump whenever sample transforms/semantics change.


def cache_path(dataset) -> Path:
    source = [(str(path.resolve()), path.stat().st_size, path.stat().st_mtime_ns) for path in dataset.shards]
    digest = hashlib.sha256(json.dumps({
        "version": CACHE_VERSION, "source": source, "horizon": dataset.horizon,
        "command_timestep": dataset.command_timestep, "height_dtype": str(dataset._height_dtype),
    }, sort_keys=True).encode())
    for item in dataset.indices:
        digest.update(f"{item.shard},{item.episode},{item.start},{int(item.collision)},{int(item.low_motion)};".encode())
    return dataset.root / ".fdm_sample_cache" / digest.hexdigest()


def open_cache(directory: Path, spec, count: int) -> dict[str, torch.Tensor]:
    manifest = json.loads((directory / "complete.json").read_text())
    if manifest != {"version": CACHE_VERSION, "count": count}:
        raise ValueError(f"Incompatible prepared sample cache: {directory}")
    storage = {}
    for name, (shape, dtype) in spec.items():
        # Copy-on-write maps are safe for torch without changing the saved files.
        array = np.load(directory / f"{name}.npy", mmap_mode="c", allow_pickle=False)
        value = torch.from_numpy(array)
        if value.shape != (count, *shape) or value.dtype != dtype:
            raise ValueError(f"Invalid prepared sample field {name} in {directory}")
        storage[name] = value
    return storage


def build_cache(dataset, directory: Path, progress) -> dict[str, torch.Tensor]:
    from itertools import groupby

    if (directory / "complete.json").is_file():
        return open_cache(directory, dataset._sample_spec(), len(dataset))
    directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = directory.with_name(f".{directory.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir()
    arrays, storage = {}, {}
    try:
        for name, (shape, dtype) in dataset._sample_spec().items():
            numpy_dtype = torch.empty((), dtype=dtype, device="cpu").numpy().dtype
            arrays[name] = np.lib.format.open_memmap(
                temporary / f"{name}.npy", mode="w+", dtype=numpy_dtype, shape=(len(dataset), *shape)
            )
            storage[name] = torch.from_numpy(arrays[name])
        for shard, requests in groupby(enumerate(dataset.indices), key=lambda pair: pair[1].shard):
            dataset._prepare_shard(shard, requests, storage, progress)
            dataset._cache.clear()
            for array in arrays.values():
                array.flush()
        storage.clear()
        arrays.clear()
        (temporary / "complete.json").write_text(json.dumps({"version": CACHE_VERSION, "count": len(dataset)}))
        try:
            os.rename(temporary, directory)
        except OSError as exc:
            # A concurrent preprocessing process may have published this key.
            if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not (directory / "complete.json").is_file():
                raise
        return open_cache(directory, dataset._sample_spec(), len(dataset))
    finally:
        dataset._cache.clear()
        storage.clear()
        arrays.clear()
        if temporary.exists():
            shutil.rmtree(temporary)
