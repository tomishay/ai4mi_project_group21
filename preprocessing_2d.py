from pathlib import Path
from typing import Tuple, Optional, Iterable
import cv2
import shutil

import numpy as np
from scipy.ndimage import median_filter
from PIL import Image


def clip_and_zscore(ct_slice, p_low, p_high) -> np.ndarray:
    
    # clip the values of ct according to percentile to erase outliers
    ct_slice = ct_slice.astype(np.float32)
    low, high = np.percentile(ct_slice, [p_low, p_high])
    ct_slice = np.clip(ct_slice, low, high)
    
    # z-score normalizatoin
    mu = float(ct_slice.mean())
    sigma = float(ct_slice.std() + 1e-8)
    ct_norm = (ct_slice - mu) / sigma

    return ct_norm


def apply_clahe(ct_slice, clip_limit=4.0, tile_grid=8) -> np.ndarray:

    # clip to avoid augmenting extreme contrast
    lo, hi = np.percentile(ct_slice, (0.5, 99.5))
    a = np.clip(ct_slice.astype(np.float32), lo, hi)
    a = (a - lo) / max(hi - lo, 1e-6)  # Schaal naar [0,1]

    # convert to [0,255] and uint8
    a8 = (a * 255.0).round().astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid, tile_grid))
    
    # make empty array and fill with CLAHE applied
    ct_clahe = clahe.apply(a8)

    # scale back to [0,1] and z-score
    ct_clahe = ct_clahe.astype(np.float32) / 255.0
    ct_clahe = (ct_clahe - ct_clahe.mean()) / (ct_clahe.std() + 1e-8)

    return ct_clahe

def to_uint8(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    x -= x.min()
    x /= (x.max() + 1e-8)
    return (x * 255.0).round().astype(np.uint8)

# ---------------------------------------------------------
def preprocess_slice(
    ct_path: Path,
    out_dir: Path,
    do_norm: bool, clip_lo: float, clip_hi: float,
    do_median: bool, median_size: int,
    do_clahe: bool, clahe_clip: float, clahe_grid: int,
):

    # load
    ct_img = np.array(Image.open(ct_path).convert('L'))
    assert ct_img.ndim == 2, f"No-2D PNG: {ct_path}"
   
    # normalisation
    if do_norm:
        ct_img = clip_and_zscore(ct_img, clip_lo, clip_hi)

    # denoising: chose median filter, because that is the most commonly used filter TODO: add source
    if do_median:
        ct_img = median_filter(ct_img, size=median_size)

    # CLAHE (per slice): this could enhance noise TODO: add source
    if do_clahe:
        ct_img = apply_clahe(ct_img, clip_limit=clahe_clip, tile_grid=clahe_grid)

    # save
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if ct_img.dtype != np.uint8:
        ct_img = to_uint8(ct_img)
    Image.fromarray(ct_img).save(out_dir)

def iter_slices(src_root) -> Iterable[Tuple[Path, Optional[Path], Path, Optional[Path]]]:
  
    # gets path relative to source root
    for split in ("train", "val"):
        img_dir = src_root / split / "img"
        gt_dir  = src_root / split / "gt"
        if not img_dir.exists():
            continue

        # go over sorted path
        for img_path in sorted(img_dir.glob("*.png")):
            rel = img_path.relative_to(src_root)
            # makes output path
            out_img_path = src_root.parent / (src_root.name + "_preproc") / rel
            # GT path
            gt_path = gt_dir / img_path.name
            out_gt_path = out_img_path.parent.parent / "gt" / img_path.name
            yield img_path, (gt_path if gt_path.exists() else None), out_img_path, (out_gt_path if gt_path.exists() else None)


def run_preprocess_slices(src_root, do_norm, norm_lo, norm_hi, do_median, median_size, do_clahe, clahe_clip, clahe_grid):
    out_root = src_root.parent / (src_root.name + "_preproc")
    out_root.mkdir(parents=True, exist_ok=True)     

    tasks = list(iter_slices(src_root))

    for img_path, gt_path, out_img_path, out_gt_path in tasks:
        out_img_path.parent.mkdir(parents=True, exist_ok=True)
        if out_gt_path:
            out_gt_path.parent.mkdir(parents=True, exist_ok=True)

        # preprocess slice
        preprocess_slice(
            img_path, out_img_path, 
            do_norm, norm_lo, norm_hi,
            do_median=do_median, median_size=median_size,
            do_clahe=do_clahe, clahe_clip=clahe_clip, clahe_grid=clahe_grid,
        )

        # copy groundtruth without preprocessing
        if gt_path and gt_path.exists():
            shutil.copy(gt_path, out_gt_path)

    print(f"Done. Preprocessed data saved under {out_root}")
    return out_root
