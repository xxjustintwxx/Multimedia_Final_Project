"""Compute the inference cost (multiply-accumulate operations) of the
dual-stream classifier directly from its layer specification.

This mirrors model/architecture.py but needs no PyTorch, so the FLOP/MAC
numbers quoted in the report are reproducible without the training stack:

    python report/evidence/compute_flops.py

Convention: one MAC = one multiply-accumulate. FLOPs are reported as 2 * MACs.
Batch normalization, ReLU, and global average pooling are element-wise and
contribute negligibly; they are omitted, as is standard for MAC accounting.
"""


def dwsep_macs(in_ch: int, out_ch: int, in_hw: int, stride: int, k: int = 3):
    """MACs of a depthwise-separable block (depthwise k*k + pointwise 1x1)."""
    out_hw = in_hw // stride
    depthwise = out_hw * out_hw * in_ch * k * k
    pointwise = out_hw * out_hw * in_ch * out_ch
    return depthwise + pointwise, out_hw


def main() -> None:
    # Image branch: five DW-separable blocks, input 64x64x3 (see ImageBranch).
    blocks = [
        # (in_ch, out_ch, stride)
        (3, 16, 2),
        (16, 32, 2),
        (32, 64, 2),
        (64, 64, 1),
        (64, 64, 1),
    ]
    hw = 64
    image_macs = 0
    for in_ch, out_ch, stride in blocks:
        macs, hw = dwsep_macs(in_ch, out_ch, hw, stride)
        image_macs += macs

    # Landmark branch MLP: 42->128->64.
    landmark_macs = 42 * 128 + 128 * 64
    # Fusion head: 128->64->6.
    fusion_macs = 128 * 64 + 64 * 6

    total_macs = image_macs + landmark_macs + fusion_macs
    print(f"image-branch MACs   : {image_macs:,}")
    print(f"landmark-branch MACs : {landmark_macs:,}")
    print(f"fusion-head MACs     : {fusion_macs:,}")
    print(f"total MACs           : {total_macs:,}  (~{total_macs/1e6:.2f} M)")
    print(f"total FLOPs (2*MACs)  : {2*total_macs:,}  (~{2*total_macs/1e6:.2f} M)")
    print(f"image-branch share   : {100*image_macs/total_macs:.1f}%")


if __name__ == "__main__":
    main()
