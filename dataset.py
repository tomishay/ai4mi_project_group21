#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025 Hoel Kervadec
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to do so, subject to the
# following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.

from pathlib import Path
import numpy as np
from typing import Callable, Union, Sequence, Optional

from torch import Tensor
from PIL import Image
from torch.utils.data import Dataset
import torch


def make_dataset(root, subset) -> list[tuple[Path, Path | None]]:
    assert subset in ['train', 'val', 'test']

    root = Path(root)
    print(f"> {root=}")

    img_path = root / subset / 'img'
    full_path = root / subset / 'gt'

    images: list[Path] = sorted(img_path.glob("*.png"))
    full_labels: list[Path | None]
    if subset != 'test':
        full_labels = sorted(full_path.glob("*.png"))
    else:
        full_labels = [None] * len(images)

    return list(zip(images, full_labels))


def make_patient_to_indices(files: list[tuple[Path, Path | None]]) -> dict[str, list[int]]:
    tmp: dict[str, list[tuple[int, int]]] = {}
    for idx, (img_path, _) in enumerate(files):
        parts = img_path.stem.split("_")
        patient_id = "_".join(parts[:-1])
        slice_number = int(parts[-1])
        tmp.setdefault(patient_id, []).append((slice_number, idx))

    patient_to_indices: dict[str, list[int]] = {}
    for patient_id, pair_list in tmp.items():
        pair_list.sort(key=lambda x: x[0])
        patient_to_indices[patient_id] = [pair[1] for pair in pair_list]
    return patient_to_indices


def _extract_patient_id(path: Path) -> str:
    stem = path.stem
    parts = stem.split('_')
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return stem


def compute_cv_split(files: Sequence[tuple[Path, Path | None]], folds: int, fold_idx: int, seed: int) -> tuple[list[int], list[int]]:
    if folds < 2:
        raise ValueError("Cross-validation requires at least two folds.")
    if not (0 <= fold_idx < folds):
        raise ValueError(f"Fold index must be in [0, {folds - 1}], got {fold_idx}.")
    if not files:
        raise ValueError('No samples available to compute a cross-validation split.')

    patient_to_indices: dict[str, list[int]] = {}
    for idx, (img_path, _) in enumerate(files):
        patient_id = _extract_patient_id(img_path)
        patient_to_indices.setdefault(patient_id, []).append(idx)

    patient_ids = list(patient_to_indices.keys())
    total_patients = len(patient_ids)
    if total_patients < folds:
        raise ValueError(f"Cross-validation requested {folds} folds, but only {total_patients} patients are available.")

    rng = np.random.default_rng(seed)
    shuffled_patients = np.array(patient_ids, dtype=object)
    rng.shuffle(shuffled_patients)

    fold_sizes = np.full(folds, total_patients // folds, dtype=int)
    fold_sizes[:total_patients % folds] += 1

    start = 0
    val_patients: list[str] = []
    for current_fold, fold_size in enumerate(fold_sizes):
        end = start + fold_size
        if current_fold == fold_idx:
            val_patients = shuffled_patients[start:end].tolist()
            break
        start = end

    val_patient_set = set(val_patients)
    train_indices: list[int] = []
    val_indices: list[int] = []

    for patient_id, idxs in patient_to_indices.items():
        if patient_id in val_patient_set:
            val_indices.extend(idxs)
        else:
            train_indices.extend(idxs)

    train_indices.sort()
    val_indices.sort()
    return train_indices, val_indices


class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment: Optional[Callable] = None, equalize=False,
                 files: list[tuple[Path, Path | None]] | None = None,
                 indices: Sequence[int] | None = None,
                 subset_label: str | None = None,
                 debug: bool = False, context: Optional[int] = None):
        if context is not None and context % 2 != 1:
            raise ValueError(f"Context must be odd or None but got {context=}")

        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation = augment
        self.equalize: bool = equalize
        self.context: int = context if context is not None else 1

        if files is not None:
            self.files = list(files)
        else:
            if subset is None:
                raise ValueError("subset must be provided when files is None")
            self.files = make_dataset(root_dir, subset)

        if indices is not None:
            self.files = [self.files[i] for i in indices]

        if debug:
            self.files = self.files[:10]

        self.subset_label = subset_label or subset or 'custom'
        self.test_mode: bool = not any(gt_path is not None for _, gt_path in self.files)
        self.patient_to_indices = make_patient_to_indices(self.files) if self.files else {}
        self.patient_ids: list[str] = sorted({_extract_patient_id(img_path) for img_path, _ in self.files})

        print(f">> Created {self.subset_label} dataset with {len(self)} images...")

    @classmethod
    def build_cv_fold(cls, root_dir, img_transform=None, gt_transform=None,
                      folds: int = 0, fold_idx: int = 0, seed: int = 13,
                      subsets: Sequence[str] = ('train', 'val'),
                      augment: Optional[Callable] = None, equalize: bool = False,
                      context: Optional[int] = None, debug: bool = False) -> tuple['SliceDataset', 'SliceDataset', int]:
        if isinstance(subsets, str):
            subsets = (subsets,)

        combined: list[tuple[Path, Path | None]] = []
        for subset_name in subsets:
            combined.extend(make_dataset(root_dir, subset_name))

        if len(combined) == 0:
            raise ValueError('No samples available to build cross-validation splits.')

        train_indices, val_indices = compute_cv_split(combined, folds, fold_idx, seed)
        train_files = [combined[i] for i in train_indices]
        val_files = [combined[i] for i in val_indices]

        train_ds = cls(subset=subsets[0] if subsets else 'train',
                       root_dir=root_dir,
                       img_transform=img_transform,
                       gt_transform=gt_transform,
                       augment=augment,
                       equalize=equalize,
                       files=train_files,
                       subset_label=f"cv-train[{fold_idx}]",
                       debug=debug,
                       context=context)

        val_ds = cls(subset=subsets[0] if subsets else 'train',
                     root_dir=root_dir,
                     img_transform=img_transform,
                     gt_transform=gt_transform,
                     augment=augment,
                     equalize=equalize,
                     files=val_files,
                     subset_label=f"cv-val[{fold_idx}]",
                     debug=debug,
                     context=context)

        total_used = len(train_files) + len(val_files)
        return train_ds, val_ds, total_used

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]

        if self.context and self.context > 1 and self.patient_to_indices:
            parts = img_path.stem.split("_")
            patient_id = "_".join(parts[:-1])
            indices = self.patient_to_indices.get(patient_id, [index])
            local_pos = indices.index(index) if index in indices else 0
            half = self.context // 2
            neighbour_global = []
            for offset in range(-half, half + 1):
                pos = max(0, min(len(indices) - 1, local_pos + offset))
                neighbour_global.append(indices[pos])
        else:
            neighbour_global = [index]

        img_tensors = [
            self.img_transform(Image.open(self.files[i][0]))
            for i in neighbour_global
        ]
        images = torch.cat(img_tensors, dim=0)

        data_dict = {"images": images,
                     "stems": img_path.stem}

        if not self.test_mode and gt_path is not None:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = images.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)

            if callable(self.augmentation):
                images_aug, gt_aug = self.augmentation(images, gt)
                data_dict["images"] = images_aug
                data_dict["gts"] = gt_aug
            else:
                data_dict["gts"] = gt

        return data_dict
