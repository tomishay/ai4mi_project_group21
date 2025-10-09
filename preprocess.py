import argparse
from pathlib import Path
from typing import Tuple, Optional
import cv2

import numpy as np
import nibabel as nib
from scipy.ndimage import zoom, median_filter
from skimage.transform import resize


def load_nifti(path: Path):
    # load nifti image
    nifti = nib.load(str(path))
    ct_img = nifti.get_fdata().astype(np.float32, copy=False)
    return ct_img, nifti.affine, nifti.header

def save_nifti(ct_img, affine, header, out_path: Path, dtype=np.float32):
    header = header.copy()
    header.set_data_dtype(dtype)

    ct_img = ct_img.astype(dtype, copy=False)
    
    nib.save(nib.Nifti1Image(ct_img, affine, header), str(out_path))


def clip_and_zscore(ct_img: np.ndarray, p_low, p_high) -> np.ndarray:
    
    # clip the values of ct according to percentile to erase outliers
    low, high = np.percentile(ct_img, [p_low, p_high])
    ct_img = np.clip(ct_img, low, high)
    
    # z-score normalizatoin
    mu = float(ct_img.mean())
    sigma = float(ct_img.std() + 1e-8)
    ct_norm = (ct_img - mu) / sigma

    return ct_norm


def resample_to_spacing(img, src_spacing, dst_spacing, order) -> np.ndarray:
    
    # resample 3d volume to asked voxel spacing
    if tuple(src_spacing) == tuple(dst_spacing):
        return img
    zoom_factors = np.array(src_spacing, dtype=np.float32) / np.array(dst_spacing, dtype=np.float32)
    return zoom(img, zoom=zoom_factors, order=order)

def fit_to_shape_with_pad(img, target_shape, order, pad_mode="constant", pad_cval=0) -> np.ndarray:
    """
    Breng volume naar target_shape (H,W,D) via:
      1) aspect-behoudende schaal in H,W (past binnen target)
      2) pad/crop tot exact target_shape
    - order=1 voor beeld, 0 voor GT.
    """
    height, width, depth = img.shape
    t_height, t_width, t_depth = target_shape

    # take smallest scale
    scale = min(t_height / height, t_width/ width)
    new_height = max(1, int(round(height * scale)))
    new_width = max(1, int(round(height * scale)))

    # resize height and width
    resized = np.empty((new_height, new_height, depth), dtype=np.float32 if order != 0 else img.dtype)
    for z in range(depth):
        resized[:,:,z] = resize(img[:,:,z], (new_height, new_width), order=order, preserve_range=True, anti_aliasing=(order > 0)).astype(resized.dtype, copy=False)

    # pad or crop to exact target heigth and target width, fill with pad_cval
    out2d = np.full((t_height, t_width, depth), pad_cval, dtype=resized.dtype)

    # offset to center image
    y0 = (t_height - new_height) // 2
    x0 = (t_width - new_width) // 2
    y1, x1 = y0 + new_height, x0 + new_width

    # if new heigth or new width is bigger than target → crop
    sy0 = max(0, -y0); sx0 = max(0, -x0)
    dy0 = max(0, y0);  dx0 = max(0, x0)
    sy1 = sy0 + min(new_height, t_height)
    sx1 = sx0 + min(new_width, t_width)
    dy1 = dy0 + min(new_height, t_width)
    dx1 = dx0 + min(new_width, t_width)

    # copy pixels to out2d
    out2d[dy0:dy1, dx0:dx1, :] = resized[sy0:sy1, sx0:sx1, :]

    # change depth to target depth
    img_out = np.full((t_height, t_width, t_depth), pad_cval, dtype=out2d.dtype)

    if depth == t_depth:
        img_out = out2d
    elif depth < t_depth:
        z0 = (t_depth - depth) // 2
        img_out[:, :, z0:z0 + depth] = out2d
    else:  # depth > t_depth
        z0 = (depth - t_depth) // 2
        img_out = out2d[:, :, z0:z0 + t_depth]

    return img_out


def apply_clahe(ct_img, clip_limit=4.0, tile_grid=8) -> np.ndarray:

    # clip to avoid augmenting extreme contrast
    lo, hi = np.percentile(ct_img, (0.5, 99.5))
    a = np.clip(ct_img.astype(np.float32), lo, hi)
    a = (a - lo) / max(hi - lo, 1e-6)  # Schaal naar [0,1]

    # convert to [0,255] and uint8
    a8 = (a * 255.0).round().astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(tile_grid, tile_grid))
    
    # make empty array and fill with CLAHE applied
    ct_clahe = np.empty_like(a8)
    for z in range(a8.shape[2]):
        ct_clahe[:, :, z] = clahe.apply(a8[:, :, z])

    # scale back to [0,1] and z-score
    ct_clahe = ct_clahe.astype(np.float32) / 255.0
    ct_clahe = (ct_clahe - ct_clahe.mean()) / (ct_clahe.std() + 1e-8)

    return ct_clahe



# ---------------------------------------------------------
def preprocess_patient(
    ct_path: Path,
    gt_path: Path,
    out_dir: Path,
    spacing_dst: Optional[Tuple[float,float,float]],
    do_norm: bool,
    clip_lo: float, clip_hi: float,
    do_clahe: bool, clahe_clip: float, clahe_grid: int,
    median_size: Tuple[int, int, int],
    target_shape: Optional[tuple[int, int, int]]
):
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # load
    ct_img, ct_aff, ct_header = load_nifti(ct_path)
    gt_img, _, gt_header = load_nifti(gt_path)
    assert ct_img.shape == gt_img.shape, f"CT/GT shape mismatch: {ct_img.shape} vs {gt_img.shape}"

    sp_src = ct_header.get_zooms()[:3]
    # print(f"[{ct_path.parent.name}] src spacing={sp_src}, shape={ct_img.shape}, "
    #       f"CT min/max={ct_img.min():.2f}/{ct_img.max():.2f}, GT uniq={np.unique(gt_img)}")

    # normalisation
    if do_norm:
        ct_img = clip_and_zscore(ct_img, clip_lo, clip_hi)

    # denoising: chose median filter, because that is the most commonly used filter TODO: add source
    if max(median_size) > 0:
        ct_img = median_filter(ct_img, size=median_size).astype(np.float32)

    # CLAHE (per slice): this could enhance noise TODO: add source
    if do_clahe:
        ct_img = apply_clahe(ct_img, clip_limit=clahe_clip, tile_grid=clahe_grid)

    # resampling: (image: order=1, GT: order=0)
    if spacing_dst is not None:
        ct_img= resample_to_spacing(ct_img, sp_src, spacing_dst, order=1)
        gt_img = resample_to_spacing(gt_img, sp_src, spacing_dst, order=0)
        
        # update header zooms
        header = ct_header.copy()
        header.set_zooms(spacing_dst + (1.0,) * max(0, len(ct_header.get_zooms()) - 3))
        ct_header = header
        gt_header = header

    if target_shape is not None:
        ct_img = fit_to_shape_with_pad(ct_img, target_shape, order=1, pad_cval=0.0)
        gt_img = fit_to_shape_with_pad(gt_img, target_shape, order=0, pad_cval=0)

    # print(f"[{ct_path.parent.name}] out shape={ct_img.shape}, "
    #         f"CT min/max={ct_img.min():.2f}/{ct_img.max():.2f}, GT uniq={np.unique(gt_img)}")

    # save
    save_nifti(ct_img, ct_aff, ct_header, out_dir / f"{ct_path.stem}_preproc.nii.gz", dtype=np.float32)
    save_nifti(gt_img, ct_aff, gt_header, out_dir / f"{gt_path.stem}_preproc.nii.gz", dtype=np.uint8)



def run_preprocess(src, dst, spacing_dst, lo, hi, median_size3, target_shape):
    dst.mkdir(parents=True, exist_ok=True)

    count = 0

    for patient_dir in src.iterdir():

        # NOTE: i am assuming GT is fixed
        ct_path = patient_dir / f"{patient_dir.name}.nii.gz"
        gt_path = patient_dir / f"GT.nii.gz"
        rel_out = dst / patient_dir.name
        # print(f"[{subset}] {patient_dir.name} → {rel_out}")

        preprocess_patient(
            ct_path=ct_path,
            gt_path=gt_path,
            out_dir=rel_out,
            spacing_dst=spacing_dst,
            clip_lo=lo, clip_hi=hi,
            do_clahe=True,          # zet True + expose via CLI indien gewenst
            clahe_clip=4.0,
            clahe_grid=8,
            do_norm=True,
            median_size=median_size3,
            target_shape=target_shape
        )
        count += 1

    print(f"Done. Processed {count} patient(s). Output: {dst}")
