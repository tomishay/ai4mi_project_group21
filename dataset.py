#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from pathlib import Path
from typing import Callable, Union, Optional

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
    tmp = {}
    for idx, (img_path, _) in enumerate(files):
        parts = img_path.stem.split("_")
        patient_id = "_".join(parts[:-1])
        slice_number = int(parts[-1])
        tmp.setdefault(patient_id, []).append((slice_number, idx))

    patient_to_indices = {}
    for patient_id, pair_list in tmp.items():
        pair_list.sort(key=lambda x: x[0])
        patient_to_indices[patient_id] = [pair[1] for pair in pair_list]
    return patient_to_indices

class SliceDataset(Dataset):
    def __init__(self, subset, root_dir, img_transform=None,
                 gt_transform=None, augment=None, equalize=False, debug=False, context=None):
        assert context % 2 == 1, f"Context must be odd or None but got {context=}"
        self.root_dir: str = root_dir
        self.img_transform: Callable = img_transform
        self.gt_transform: Callable = gt_transform
        self.augmentation: Optional[Callable] = augment
        self.equalize: bool = equalize

        self.test_mode: bool = subset == 'test'
        self.context: int = context

        self.files = make_dataset(root_dir, subset)
        if debug:
            self.files = self.files[:10]
        
        self.patient_to_indices = make_patient_to_indices(self.files)

        print(f">> Created {subset} dataset with {len(self)} images...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index) -> dict[str, Union[Tensor, int, str]]:
        img_path, gt_path = self.files[index]
        if self.context and self.context > 1:
            # determine patient id
            parts = img_path.stem.split("_")
            patient_id = "_".join(parts[:-1])
            indices = self.patient_to_indices[patient_id]
            # find our position inside the patient's slice list
            local_pos = indices.index(index)
            half = self.context // 2
            # collect global indices of neighbouring slices
            neighbour_global = []
            for offset in range(-half, half + 1):
                pos = local_pos + offset
                # clamp to valid range
                pos = max(0, min(len(indices) - 1, pos))
                neighbour_global.append(indices[pos])

        else:
            neighbour_global = [index]

        img_tensors = [
                self.img_transform(Image.open(self.files[i][0]))
                for i in neighbour_global
            ]
        # stack into a single tensor of shape (C, H, W)
        images = torch.cat(img_tensors, dim=0)
        data_dict = {"images": images, "stems": img_path.stem}
        
        if not self.test_mode:
            gt: Tensor = self.gt_transform(Image.open(gt_path))

            _, W, H = images.shape
            K, _, _ = gt.shape
            assert gt.shape == (K, W, H)
            
            if self.augmentation is not None:
                images_aug, gt_aug = self.augmentation(images, gt)
                data_dict["images"] = images_aug
                data_dict["gts"] = gt_aug
            else:
                data_dict["gts"] = gt

              
        


        return data_dict

