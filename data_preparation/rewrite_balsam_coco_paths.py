#!/usr/bin/env python3
"""Rewrite Balsam COCO image/DSM paths from the Windows root to Linux.

By default, Windows runs target the local dataset annotations; Linux runs target
the requested remote dataset location. The script is dry-run unless --apply is
given. On apply, original JSON files are saved as ``<name>.json.bak`` first.
"""

from __future__ import annotations

import argparse
import json
import os
import posixpath
import shutil
import tempfile
from pathlib import Path, PureWindowsPath


WINDOWS_DATA_ROOT = (
    r"D:\KEYAN\TreeMonitor\Data\DatasetNew\processed\balsam_hsi_1024_stride256"
)
LINUX_DATA_ROOT = (
    "/home/huiying/Documents/ITCsegmentation/Data/DatasetNew/processed/"
    "balsam_hsi_1024_stride256"
)


def rewrite_path(value: str, old_root: str, new_root: str) -> str:
    """Map a path below old_root to new_root, preserving its relative suffix."""
    # COCO paths were serialized on Windows, but tolerate either slash style.
    normalized = value.replace("/", "\\")
    old_normalized = old_root.replace("/", "\\").rstrip("\\")
    prefix = old_normalized + "\\"
    if normalized.casefold().startswith(prefix.casefold()):
        suffix = PureWindowsPath(normalized[len(prefix) :]).as_posix()
        return posixpath.join(new_root.rstrip("/"), suffix)

    # Allow repeated runs after the JSON has already been converted.
    if value.replace("\\", "/").startswith(new_root.rstrip("/") + "/"):
        return value.replace("\\", "/")

    raise ValueError(f"Path is outside the expected data roots: {value!r}")


def main() -> int:
    if os.name == "nt":
        default_annotation_dir = Path(WINDOWS_DATA_ROOT) / "annotations"
    else:
        default_annotation_dir = Path(LINUX_DATA_ROOT) / "annotations"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--annotation-dir",
        type=Path,
        default=default_annotation_dir,
        help=f"Folder containing COCO JSON files (default: {default_annotation_dir})",
    )
    parser.add_argument("--old-root", default=WINDOWS_DATA_ROOT)
    parser.add_argument("--new-root", default=LINUX_DATA_ROOT)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write converted JSON files and preserve originals as .json.bak",
    )
    args = parser.parse_args()

    json_files = sorted(args.annotation_dir.glob("*.json"))
    if not json_files:
        parser.error(f"No JSON files found in {args.annotation_dir}")

    converted: list[tuple[Path, str, int]] = []
    for json_path in json_files:
        document = json.loads(json_path.read_text(encoding="utf-8-sig"))
        changed = 0
        for image in document.get("images", []):
            for key in ("file_name", "dsm_path"):
                value = image.get(key)
                if value is None:
                    continue
                if not isinstance(value, str):
                    raise TypeError(f"{json_path}: images[{image.get('id')}].{key} is not a string")
                updated = rewrite_path(value, args.old_root, args.new_root)
                if updated != value:
                    image[key] = updated
                    changed += 1
        converted.append((json_path, json.dumps(document, ensure_ascii=False, indent=2) + "\n", changed))

    for json_path, output_text, changed in converted:
        print(f"{json_path.name}: {changed} path(s) to rewrite")
        if not args.apply or changed == 0:
            continue

        backup = json_path.with_suffix(json_path.suffix + ".bak")
        if not backup.exists():
            shutil.copy2(json_path, backup)
            print(f"  backup: {backup}")
        fd, temporary_name = tempfile.mkstemp(
            prefix=json_path.name + ".", suffix=".tmp", dir=json_path.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as stream:
                stream.write(output_text)
            os.replace(temporary_name, json_path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    if not args.apply:
        print("Dry run only. Re-run with --apply to write changes (backups are created).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
