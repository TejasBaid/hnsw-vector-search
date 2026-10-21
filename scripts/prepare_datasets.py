#!/usr/bin/env python3
"""Download ann-benchmarks datasets and convert them to .fvecs/.ivecs.

Usage:
    python scripts/prepare_datasets.py --data-dir data [--datasets sift glove]

The HDF5 files come from http://ann-benchmarks.com and contain `train`
(base vectors), `test` (queries) and `neighbors` (exact top-100 ids).
"""
from __future__ import annotations

import argparse
import pathlib
import urllib.request

import h5py
import numpy as np

DATASETS = {
    "sift": "sift-128-euclidean.hdf5",
    "glove": "glove-100-angular.hdf5",
}
BASE_URL = "http://ann-benchmarks.com/"


def write_vecs(path: pathlib.Path, mat: np.ndarray, dtype) -> None:
    mat = np.ascontiguousarray(mat, dtype=dtype)
    n, d = mat.shape
    out = np.empty((n, d + 1), dtype=np.int32)
    out[:, 0] = d
    out[:, 1:] = mat.view(np.int32)
    out.tofile(path)


def prepare(name: str, data_dir: pathlib.Path) -> None:
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    h5_path = raw_dir / DATASETS[name]
    if not h5_path.exists():
        print(f"downloading {BASE_URL + DATASETS[name]} ...")
        urllib.request.urlretrieve(BASE_URL + DATASETS[name], h5_path)
    out_dir = data_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(h5_path, "r") as f:
        train = np.asarray(f["train"], dtype=np.float32)
        test = np.asarray(f["test"], dtype=np.float32)
        neighbors = np.asarray(f["neighbors"], dtype=np.int32)
        metric = f.attrs.get("distance", "?")
    write_vecs(out_dir / "base.fvecs", train, np.float32)
    write_vecs(out_dir / "query.fvecs", test, np.float32)
    write_vecs(out_dir / "groundtruth.ivecs", neighbors, np.int32)
    print(f"{name}: base={train.shape} query={test.shape} gt={neighbors.shape} metric={metric}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=pathlib.Path, default=pathlib.Path("data"))
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    args = ap.parse_args()
    for name in args.datasets:
        prepare(name, args.data_dir)


if __name__ == "__main__":
    main()
