#!/usr/bin/env python3
"""
After download_dataset.sh extracts the images, run this to:
  1. Print the actual folder structure (so you can verify extraction worked)
  2. Subsample each class to MAX_PER_CLASS images (deletes the rest)
  3. Print a final count per class

Usage:
    python subsample_dataset.py [--images_dir ./hagrid_data/images] [--max 4000] [--dry-run]
"""

import argparse
import os
import random
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir", default="./hagrid_data/images")
    parser.add_argument("--max", type=int, default=4000, help="Max images to keep per class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true", help="Print what would be deleted without deleting")
    args = parser.parse_args()

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        print(f"ERROR: {images_dir} does not exist. Run download_dataset.sh first.")
        return

    print(f"=== Scanning {images_dir} ===")
    class_dirs = sorted([d for d in images_dir.iterdir() if d.is_dir()])

    if not class_dirs:
        print("No subdirectories found. Checking one level deeper...")
        class_dirs = sorted([d for d in images_dir.glob("*/*") if d.is_dir()])

    if not class_dirs:
        print("Still no class folders found. Listing what's in the directory:")
        for p in images_dir.iterdir():
            print(f"  {p}")
        return

    random.seed(args.seed)
    total_kept = 0
    total_deleted = 0

    for class_dir in class_dirs:
        images = sorted(class_dir.glob("*.jpg")) + sorted(class_dir.glob("*.png")) + \
                 sorted(class_dir.glob("*.jpeg")) + sorted(class_dir.glob("*.JPG"))

        if not images:
            print(f"  {class_dir.name}: 0 images found (wrong format or empty)")
            continue

        n = len(images)
        keep = args.max

        if n <= keep:
            print(f"  {class_dir.name}: {n} images — keeping all (under limit)")
            total_kept += n
            continue

        random.shuffle(images)
        to_keep = set(images[:keep])
        to_delete = [img for img in images if img not in to_keep]

        if args.dry_run:
            print(f"  {class_dir.name}: {n} images — would keep {keep}, delete {len(to_delete)}")
        else:
            for img in to_delete:
                img.unlink()
            print(f"  {class_dir.name}: {n} → kept {keep}, deleted {len(to_delete)}")

        total_kept += keep
        total_deleted += len(to_delete)

    print(f"\n=== Summary ===")
    print(f"  Total kept:   {total_kept:,}")
    print(f"  Total deleted: {total_deleted:,}")
    if args.dry_run:
        print("  (dry run — nothing was actually deleted)")


if __name__ == "__main__":
    main()
