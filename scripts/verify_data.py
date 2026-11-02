"""Verify that every packaged input matches its recorded SHA-256 fingerprint."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def verify(include_supplementary=True):
    manifest = json.loads((ROOT / "data/manifest.json").read_text())
    checked = 0
    for record in manifest["files"]:
        if not include_supplementary and not record["required_for_report"]:
            continue
        path = ROOT / record["path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing {record['path']}; run git lfs pull after cloning.")
        if path.stat().st_size != record["bytes"]:
            raise ValueError(
                f"Unexpected size for {record['path']}; check for an unresolved Git LFS pointer."
            )
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"Checksum mismatch: {record['path']}")
        if "original_sha256" in record:
            original = hashlib.sha256()
            original_bytes = 0
            with gzip.open(path, "rb") as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b""):
                    original.update(block)
                    original_bytes += len(block)
            if (
                original.hexdigest() != record["original_sha256"]
                or original_bytes != record["original_bytes"]
            ):
                raise ValueError(f"Decompressed original does not match: {record['path']}")
        checked += 1
    print(f"Verified {checked} input files.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--required-only", action="store_true")
    verify(not parser.parse_args().required_only)
