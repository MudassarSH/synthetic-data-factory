"""Create disjoint clean source sets before synthesis."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from .dataset import find_images


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_clean_pages(source_dir: str | Path, output_dir: str | Path, *, validation_fraction=0.18, seed=0, group_by="parent") -> dict:
    """Copy source documents into train/validation, grouping duplicate bytes.

    Use group_by='parent' for multi-page documents stored in per-document
    subfolders. A source file is copied exactly once and the split is recorded.
    """
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if source_dir == output_dir or source_dir in output_dir.parents or output_dir in source_dir.parents:
        raise ValueError("source and output directories must be separate")
    if not 0 < validation_fraction < 1 or seed < 0 or group_by not in {"file", "parent"}:
        raise ValueError("invalid validation_fraction, seed, or group_by")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    paths = find_images(source_dir)
    if len(paths) < 2:
        raise ValueError("at least two clean source pages are required")

    # Union pages in the same document folder or with identical source bytes.
    parents = list(range(len(paths)))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left, right):
        parents[root(right)] = root(left)

    seen = {}
    hashes = []
    for index, path in enumerate(paths):
        relative = path.relative_to(source_dir)
        checksum = _sha256(path)
        hashes.append(checksum)
        keys = [("content", checksum)]
        if group_by == "parent" and len(relative.parts) > 1:
            keys.append(("document", relative.parts[0]))
        for key in keys:
            if key in seen:
                union(index, seen[key])
            else:
                seen[key] = index

    groups = {}
    for index in range(len(paths)):
        groups.setdefault(root(index), []).append(index)
    if len(groups) < 2:
        raise ValueError("all pages belong to one source group; cannot make a disjoint split")

    scored = []
    for members in groups.values():
        identity = "|".join(sorted(paths[i].relative_to(source_dir).as_posix() for i in members))
        score = int.from_bytes(hashlib.sha256(f"{seed}\0{identity}".encode()).digest()[:8], "big") / 2**64
        scored.append((score, members))
    scored.sort(key=lambda item: item[0])
    target_pages = validation_fraction * len(paths)
    best_cutoff, best_error, pages_so_far = 1, float("inf"), 0
    for cutoff, (_, members) in enumerate(scored[:-1], start=1):
        pages_so_far += len(members)
        error = abs(pages_so_far - target_pages)
        if error < best_error:
            best_cutoff, best_error = cutoff, error
    assignments = {
        index: ("validation" if position < best_cutoff else "train")
        for position, (_, members) in enumerate(scored)
        for index in members
    }

    records = []
    for index, path in enumerate(paths):
        relative = path.relative_to(source_dir)
        split = assignments[index]
        destination = output_dir / split / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
        records.append({"source": relative.as_posix(), "split": split, "sha256": hashes[index]})
    manifest = {"seed": seed, "validation_fraction": validation_fraction, "group_by": group_by, "counts": {name: sum(row["split"] == name for row in records) for name in ("train", "validation")}, "pages": records}
    (output_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_dir")
    parser.add_argument("output_dir")
    parser.add_argument("--validation-fraction", type=float, default=0.18)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-by", choices=("file", "parent"), default="parent")
    args = parser.parse_args()
    manifest = split_clean_pages(args.source_dir, args.output_dir, validation_fraction=args.validation_fraction, seed=args.seed, group_by=args.group_by)
    print(json.dumps(manifest["counts"], indent=2))


if __name__ == "__main__":
    main()
