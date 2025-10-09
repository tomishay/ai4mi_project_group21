#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

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

import argparse
import warnings
import json
import math
from typing import Any
from pathlib import Path
from pprint import pprint
from shutil import copytree, rmtree

import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, Tensor
from torch.utils.data import DataLoader

from functools import partial 

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)

from losses import (CrossEntropy)

datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}

def img_transform(img):
        img = img.convert('L')
        img = np.array(img)[np.newaxis, ...]
        img = img / 255  # max <= 1
        img = torch.tensor(img, dtype=torch.float32)
        return img

def gt_transform(K, img):
        img = np.array(img)[...]
        # The idea is that the classes are mapped to {0, 255} for binary cases
        # {0, 85, 170, 255} for 4 classes
        # {0, 51, 102, 153, 204, 255} for 6 classes
        # Very sketchy but that works here and that simplifies visualization
        img = img / (255 / (K - 1)) if K != 5 else img / 63  # max <= 1
        img = torch.tensor(img, dtype=torch.int64)[None, ...]  # Add one dimension to simulate batch
        img = class2one_hot(img, K=K)
        return img[0]


def _tensor_mean_or_nan(tensor: Tensor) -> float:
    if tensor.numel() == 0:
        return float('nan')
    return tensor.mean().item()


def setup(args) -> tuple[nn.Module, Any, Any, DataLoader, DataLoader, int]:
    # Networks and scheduler
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    net = datasets_params[args.dataset]['net'](1, K, kernels=kernels, factor=factor)
    net.init_weights()
    net.to(device)

    lr = 0.0005
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))

    # Dataset part
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path("data") / args.dataset

    if args.cv_folds and args.cv_folds > 1:
        train_set, val_set, total_samples = SliceDataset.build_cv_fold(
            root_dir=root_dir,
            img_transform=img_transform,
            gt_transform=partial(gt_transform, K),
            folds=args.cv_folds,
            fold_idx=args.cv_index,
            seed=args.cv_seed,
        )
        total_patients = len(set(train_set.patient_ids) | set(val_set.patient_ids))
        print(f">> Cross-validation fold {args.cv_index + 1}/{args.cv_folds}: {len(train_set)} train / {len(val_set)} val slices (patients: {len(train_set.patient_ids)} train / {len(val_set.patient_ids)} val, total={total_patients})")
    else:
        train_set = SliceDataset('train',
                                 root_dir,
                                 img_transform=img_transform,
                                 gt_transform=partial(gt_transform, K))
        val_set = SliceDataset('val',
                               root_dir,
                               img_transform=img_transform,
                               gt_transform=partial(gt_transform, K))
    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=4,
                              shuffle=True)

    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=4,
                            shuffle=False)

    args.dest.mkdir(parents=True, exist_ok=True)

    return (net, optimizer, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode}")
    net, optimizer, device, train_loader, val_loader, K = setup(args)

    if args.mode == "full":
        loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
    elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
        loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
    else:
        raise ValueError(args.mode, args.dataset)

    # Notice one has the length of the _loader_, and the other one of the _dataset_
    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0.0
    best_epoch: int = -1

    for e in range(args.epochs):
        for m in ['train', 'val']:
            match m:
                case 'train':
                    net.train()
                    opt = optimizer
                    cm = Dcm
                    desc = f">> Training   ({e: 4d})"
                    loader = train_loader
                    log_loss = log_loss_tra
                    log_dice = log_dice_tra
                case 'val':
                    net.eval()
                    opt = None
                    cm = torch.no_grad
                    desc = f">> Validation ({e: 4d})"
                    loader = val_loader
                    log_loss = log_loss_val
                    log_dice = log_dice_val

            with cm():  # Either dummy context manager, or the torch.no_grad for validation
                j = 0
                tq_iter = tqdm_(enumerate(loader), total=len(loader), desc=desc)
                for i, data in tq_iter:
                    img = data['images'].to(device)
                    gt = data['gts'].to(device)

                    if opt:  # So only for training
                        opt.zero_grad()

                    # Sanity tests to see we loaded and encoded the data correctly
                    assert 0 <= img.min() and img.max() <= 1
                    B, _, W, H = img.shape

                    pred_logits = net(img)
                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg, gt)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()

                    if m == 'val':
                        with warnings.catch_warnings():
                            warnings.filterwarnings('ignore', category=UserWarning)
                            predicted_class: Tensor = probs2class(pred_probs)
                            mult: int = 63 if K == 5 else (255 / (K - 1))
                            save_images(predicted_class * mult,
                                        data['stems'],
                                        args.dest / f"iter{e:03d}" / m)

                    j += B  # Keep in mind that _in theory_, each batch might have a different size
                    # For the DSC average: do not take the background class (0) into account:
                    postfix_dict: dict[str, str] = {"Dice": f"{log_dice[e, :j, 1:].mean():05.3f}",
                                                    "Loss": f"{log_loss[e, :i + 1].mean():5.2e}"}
                    if K > 2:
                        postfix_dict |= {f"Dice-{k}": f"{log_dice[e, :j, k].mean():05.3f}"
                                         for k in range(1, K)}
                    tq_iter.set_postfix(postfix_dict)

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

        current_dice: float = _tensor_mean_or_nan(log_dice_val[e, :, 1:])
        if not math.isnan(current_dice) and current_dice > best_dice:
            previous_best = best_dice if best_epoch >= 0 else 0.0
            message = f">>> Improved dice at epoch {e}: {previous_best:05.3f}->{current_dice:05.3f} DSC"
            print(message)
            best_dice = current_dice
            best_epoch = e
            with open(args.dest / "best_epoch.txt", 'w') as f:
                f.write(message)

            best_folder = args.dest / "best_epoch"
            if best_folder.exists():
                rmtree(best_folder)
            copytree(args.dest / f"iter{e:03d}", Path(best_folder))

            torch.save(net, args.dest / "bestmodel.pkl")
            torch.save(net.state_dict(), args.dest / "bestweights.pt")

    final_epoch_idx = max(args.epochs - 1, 0)
    final_val_dice = _tensor_mean_or_nan(log_dice_val[final_epoch_idx, :, 1:])
    final_val_loss = _tensor_mean_or_nan(log_loss_val[final_epoch_idx])

    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    train_patients = list(getattr(train_dataset, "patient_ids", []))
    val_patients = list(getattr(val_dataset, "patient_ids", []))

    summary = {
        "fold_index": getattr(args, "cv_index", None),
        "dest": args.dest,
        "best_epoch": best_epoch if best_epoch >= 0 else None,
        "best_dice": float(best_dice if best_epoch >= 0 else float('nan')),
        "final_val_dice": final_val_dice,
        "final_val_loss": final_val_loss,
        "epochs": args.epochs,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_patient_count": len(train_patients),
        "val_patient_count": len(val_patients),
        "train_patients": train_patients,
        "val_patients": val_patients,
    }

    return summary

def summarize_crossval_runs(base_dest: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return

    metric_keys = ['best_dice', 'final_val_dice', 'final_val_loss']
    aggregated: dict[str, dict[str, float]] = {}
    for key in metric_keys:
        values: list[float] = []
        for summary in summaries:
            value = summary.get(key)
            if value is None:
                continue
            value = float(value)
            if math.isnan(value):
                continue
            values.append(value)
        if values:
            aggregated[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
            }
        else:
            aggregated[key] = {
                'mean': float('nan'),
                'std': float('nan'),
            }

    serialised = []
    for summary in summaries:
        serial = dict(summary)
        if 'dest' in serial:
            serial['dest'] = str(serial['dest'])
        serialised.append(serial)

    base_dest.mkdir(parents=True, exist_ok=True)
    summary_path = base_dest / 'cv_summary.json'
    payload = {'fold_results': serialised, 'aggregated': aggregated}
    summary_path.write_text(json.dumps(payload, indent=2))

    print('>>> Cross-validation summary')
    for metric, stats in aggregated.items():
        mean = stats['mean']
        std = stats['std']
        if math.isnan(mean):
            print(f"    {metric}: mean=nan std=nan (insufficient data)")
        else:
            print(f"    {metric}: mean={mean:0.4f} std={std:0.4f}")
    print(f"    Saved summary to {summary_path}")



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save the results (predictions and weights).")

    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--cv-folds', type=int, default=0,
                        help="Number of folds for cross-validation; set >1 to enable.")
    parser.add_argument('--cv-index', type=int, default=0,
                        help="Fold index to train when cross-validation is enabled.")
    parser.add_argument('--cv-run-all', action='store_true',
                        help="Train sequentially on every fold when cross-validation is enabled.")
    parser.add_argument('--cv-seed', type=int, default=42,
                        help="Random seed used to shuffle samples before fold splits.")

    args = parser.parse_args()

    if args.cv_folds < 0:
        raise ValueError("--cv-folds must be >= 0")

    use_crossval = args.cv_folds and args.cv_folds > 1
    if args.cv_run_all and not use_crossval:
        warnings.warn("--cv-run-all ignored because --cv-folds <= 1")

    base_dest = args.dest

    if use_crossval:
        if args.cv_run_all :
            summaries: list[dict[str, Any]] = []
            for fold_idx in range(args.cv_folds):
                fold_args = argparse.Namespace(**vars(args))
                fold_args.cv_index = fold_idx
                fold_args.cv_run_all = False
                fold_args.dest = base_dest / f"fold{fold_idx:02d}"
                pprint(fold_args)
                summary = runTraining(fold_args)
                if summary:
                    summaries.append(summary)
            summarize_crossval_runs(base_dest, summaries)
        else:
            if not (0 <= args.cv_index < args.cv_folds):
                raise ValueError(f"--cv-index must be in [0, {args.cv_folds - 1}] when cross-validation is enabled")
            args.dest = base_dest / f"fold{args.cv_index:02d}"
            pprint(args)
            runTraining(args)
    else:
        args.dest = base_dest
        pprint(args)
        runTraining(args)


if __name__ == '__main__':
    main()
