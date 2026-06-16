#!/usr/bin/env bash
# Downloads the HaGRIDv2 512px dataset (119.4 GB) and extracts only the
# gesture classes needed for the project, then removes the zip.
#
# Disk required: ~155 GB during download, ~36 GB after cleanup.
# Run on a server with good bandwidth; uses wget -c so it's resumable.
#
# Usage:
#   bash download_dataset.sh [save_dir]
#   default save_dir: ./hagrid_data

set -euo pipefail

SAVE_DIR="${1:-./hagrid_data}"
V2_BASE="https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2"

# 5 target gesture classes (labels 1-5)
TARGET_CLASSES=(fist like ok one palm)

# N/A gesture classes — diverse visually confusable ones + no_gesture (only 0.49 GB)
# Excluded: two-handed gestures (hands_heart2, etc.) since TAs exclude them from N/A test
NA_CLASSES=(dislike peace stop call no_gesture)

ALL_CLASSES=("${TARGET_CLASSES[@]}" "${NA_CLASSES[@]}")

mkdir -p "$SAVE_DIR"
cd "$SAVE_DIR"

echo "=== Step 1: Download annotations (landmarks + bboxes for all classes) ==="
if [ ! -f "annotations.zip" ]; then
    wget -c "${V2_BASE}/annotations_with_landmarks/annotations.zip" -O annotations.zip
else
    echo "annotations.zip already exists, skipping."
fi

echo "=== Extracting annotations ==="
unzip -q -o annotations.zip -d annotations/ && rm -f annotations.zip
echo "Annotations extracted."

echo ""
echo "=== Step 2: Download 512px combined dataset (119.4 GB) ==="
echo "    This is resumable — safe to Ctrl-C and re-run."
if [ ! -f "hagridv2_512.zip" ]; then
    wget -c "${V2_BASE}/hagridv2_512.zip" -O hagridv2_512.zip
else
    echo "hagridv2_512.zip already present, skipping download."
fi

echo ""
echo "=== Step 3: Selectively extract needed gesture classes ==="
echo "    Classes: ${ALL_CLASSES[*]}"

mkdir -p images

for cls in "${ALL_CLASSES[@]}"; do
    echo "  Extracting: $cls"
    # List entries in the zip that belong to this class folder, then extract them.
    # The 512px zip uses folder names matching gesture names directly.
    unzip -q -o hagridv2_512.zip "${cls}/*" -d images/ 2>/dev/null || \
    unzip -q -o hagridv2_512.zip "*/${cls}/*" -d images/ 2>/dev/null || \
    echo "  WARNING: could not find '${cls}' folder in zip — check zip structure below."
done

echo ""
echo "=== Checking extracted structure ==="
ls -lh images/

echo ""
echo "=== Step 4: Delete the large zip to free disk space ==="
read -rp "Delete hagridv2_512.zip now to free ~120 GB? [y/N] " confirm
if [[ "${confirm,,}" == "y" ]]; then
    rm -f hagridv2_512.zip
    echo "Deleted."
else
    echo "Kept. Remember to delete it manually when done."
fi

echo ""
echo "=== Done! ==="
echo "Dataset saved to: ${SAVE_DIR}/images/"
echo "Annotations at:   ${SAVE_DIR}/annotations/"
echo ""
echo "Next: run subsample_dataset.py to cap each class at ~4000 images."
