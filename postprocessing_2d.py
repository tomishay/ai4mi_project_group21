from pathlib import Path
import numpy as np
from skimage.measure import label as cc_label
from PIL import Image
from typing import Dict, Optional


# 2D Connected Component Analysis

def cc2d_segthor(classmap, K=5, keep_k: Optional[dict] = None, min_size: Optional[dict] = None, connectivity=2) -> np.ndarray:
    """
    Perform per-class 2D Connected Component Analysis (CCA) filtering on a single segmentation slice.
    """
    # sensible defaults
    if keep_k is None:
        keep_k = {c: 1 for c in range(1, K)}

    if min_size is None:
        # default pixel thresholds per class for SEGTHOR
        if K == 5:
            min_size = {1: 80, 2: 150, 3: 80, 4: 80}  
        else: 
            {c: 50 for c in range(1, K)}

    out = classmap.copy()
    for c in range(1, K):  # skip background (0)
        mask = (out == c)
        if not mask.any():
            continue

        # label connecting components
        lab = cc_label(mask, connectivity=connectivity)
        if lab.max() == 0:
            out[mask] = 0
            continue
        
        # compute pixel count and keep only those that are bigger than min_size
        counts = np.bincount(lab.ravel())
        valid_ids = [cid for cid in range(1, len(counts)) if counts[cid] >= min_size.get(c, 0)]
        if not valid_ids:
            out[mask] = 0
            continue

        # sort the groups and keep top-k
        valid_ids = np.array(valid_ids)
        sizes = counts[valid_ids]
        k = max(1, keep_k.get(c, 1))
        keep_ids = valid_ids[np.argsort(sizes)[::-1][:k]]

        # build mask and only keep that one
        keep_mask = np.isin(lab, keep_ids)
        out[mask] = 0
        out[keep_mask] = c

    return out

# run per batch (used)

def clean_pred_batch_2d(
    pred_batch, K=5, keep_k: Optional[dict] = None, min_size: Optional[dict] = None, connectivity=2,) -> np.ndarray:
    """
    In-memory CCA cleanup for a batch of predicted class maps.
    """
    assert pred_batch.ndim == 3, f"Expected (B,H,W), got {pred_batch.shape}"
    B, H, W = pred_batch.shape
    out = np.empty_like(pred_batch)
    for b in range(B):
        out[b] = cc2d_segthor(pred_batch[b], K=K, keep_k=keep_k, min_size=min_size,connectivity=connectivity)
    return out


# Folder-level post-processing utility (not used)

def run_postprocess(pred_dir, out_dir, K=5, keep_k: Optional[dict] = None, min_size: Optional[dict] = None, overwrite=False,) -> None:
    """
    Apply 2D connected-component filtering to all PNG prediction masks in a folder.
    """
    
    # get pngs
    pngs = sorted(pred_dir.glob("*.png"))
    if not pngs:
        print(f"[WARN] No PNG files found in {pred_dir}")
        return


    if not overwrite:
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = pred_dir

    print(f">> Running 2D Connected Components filtering on {len(pngs)} PNGs in {pred_dir}")

    for p in pngs:
        # load image and convert to label map
        u8 = np.array(Image.open(p).convert("L"))

        # turn png into labels
        if K == 5:
            lbl = np.rint(u8.astype(np.float32) / 63.0).astype(np.uint8)
            lbl = np.clip(lbl, 0, 4)
        else:
            step = 255.0 / (K - 1)
            lbl = np.rint(u8.astype(np.float32) / step).astype(np.uint8)
            lbl = np.clip(lbl, 0, K - 1)


        # apply CCA-based cleanup
        lbl_cc = cc2d_segthor(lbl, K=K, keep_k=keep_k, min_size=min_size)

        # convert back to PNG grayscale encoding
        if K == 5:
            u8_cc = (lbl_cc.astype(np.uint8) * 63)
        else:
            step = 255.0 / (K - 1)
            u8_cc = np.rint(lbl_cc.astype(np.float32) * step).astype(np.uint8)

        Image.fromarray(u8_cc).save(out_dir / p.name)

    print(f">> Done. {len(pngs)} slices processed.")
    if not overwrite:
        print(f">> Cleaned PNGs saved in: {out_dir}")