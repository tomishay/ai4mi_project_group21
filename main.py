#!/usr/bin/env python3

# MIT License

# Copyright (c) 2025 Hoel Kervadec, Caroline Magg

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to do so, subject to the
# following conditions:

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
import csv
from typing import Any
from pathlib import Path
from pprint import pprint
from operator import itemgetter
from shutil import copytree, rmtree

import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch import nn, Tensor
from torchvision import transforms
from torch.utils.data import DataLoader

from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import OneCycleLR
from lion_pytorch import Lion
from itertools import product

from dataset import SliceDataset
from ShallowNet import shallowCNN
from ENet import ENet
from ENet_enhance import ENet_enhance
from utils import (Dcm,
                   class2one_hot,
                   probs2one_hot,
                   probs2class,
                   tqdm_,
                   dice_coef,
                   save_images)
from vit_seg import TinyViTSeg
from losses import CrossEntropy
from new_losses import CombinedLoss
from augment import OnlineAugment2D, AugConfig2D
from preprocessing_2d import run_preprocess_slices


datasets_params: dict[str, dict[str, Any]] = {}
# K for the number of classes
# Avoids the classes with C (often used for the number of Channel)
datasets_params["TOY2"] = {'K': 2, 'net': shallowCNN, 'B': 2, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}
datasets_params["SEGTHOR_CLEAN_preproc"] = {'K': 5, 'net': ENet, 'B': 8, 'kernels': 8, 'factor': 2}


def _tensor_mean_or_nan(tensor: Tensor) -> float:
    if tensor.numel() == 0:
        return float('nan')
    return tensor.mean().item()


def setup(args) -> tuple[nn.Module, Any, Any, Any, DataLoader, DataLoader, int]:
    gpu: bool = args.gpu and torch.cuda.is_available()
    device = torch.device("cuda") if gpu else torch.device("cpu")
    print(f">> Picked {device} to run experiments")

    K: int = datasets_params[args.dataset]['K']
    kernels: int = datasets_params[args.dataset]['kernels'] if 'kernels' in datasets_params[args.dataset] else 8
    factor: int = datasets_params[args.dataset]['factor'] if 'factor' in datasets_params[args.dataset] else 2
    if args.arch == 'enetx':
        net = ENet_enhance(in_dim=args.context, out_dim=K,
                           kernels=datasets_params[args.dataset].get('kernels', 8),
                           factor=datasets_params[args.dataset].get('factor', 2),
                           use_se=True, return_aux=True)

    elif args.arch == 'vit':
        net = TinyViTSeg(in_dim=args.context, out_dim=K,
                         embed_dim=192, depth=6, heads=6, patch=16, drop=0.0)
    else:
        net = datasets_params[args.dataset]['net'](args.context, K, kernels=kernels, factor=factor)

    net.init_weights()
    net.to(device)

    # --- Optimizer selection ---
    lr = 0.0005
    optimizer_type = args.optimizer if hasattr(args, 'optimizer') else 'adamw'
    if optimizer_type == 'adamw':
        optimizer = AdamW(net.parameters(), lr=lr, betas=(0.9, 0.999))
    elif optimizer_type == 'adam':
        optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    elif optimizer_type == 'radam':
        from torch.optim import RAdam
        optimizer = RAdam(net.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        optimizer = SGD(net.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    elif optimizer_type == 'lion':
        optimizer = Lion(net.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    # Dataset setup
    B: int = datasets_params[args.dataset]['B']
    root_dir = Path("data") / args.dataset

    img_transform = transforms.Compose([
        lambda img: img.convert('L'),
        lambda img: np.array(img)[np.newaxis, ...],
        lambda nd: nd / 255,
        lambda nd: torch.tensor(nd, dtype=torch.float32)
    ])

    gt_transform = transforms.Compose([
        lambda img: np.array(img)[...],
        lambda nd: nd / (255 / (K - 1)) if K != 5 else nd / 63,
        lambda nd: torch.tensor(nd, dtype=torch.int64)[None, ...],
        lambda t: class2one_hot(t, K=K),
        itemgetter(0)
    ])

    # --- Augmentation config ---
    if args.aug == 'online':
        aug_cfg = AugConfig2D(
            rot_deg=8.0, shear_deg=5.0, translate=0.010, p_rot90=0.10,
            p_roi_focus=0.60, small_class_indices=(0, 2),
            p_elastic=0.20, elastic_sigma=8.0, elastic_alpha=1.2
        )
        aug = OnlineAugment2D(aug_cfg)
    else:
        aug = None

    use_crossval = getattr(args, "cv_folds", 0) and args.cv_folds > 1

    if use_crossval:
        train_set, val_set, _ = SliceDataset.build_cv_fold(
            root_dir=root_dir,
            img_transform=img_transform,
            gt_transform=gt_transform,
            folds=args.cv_folds,
            fold_idx=args.cv_index,
            seed=args.cv_seed,
            augment=None,
            context=args.context,
            debug=getattr(args, "debug", False)
        )
        train_set.augmentation = aug
        val_set.augmentation = None
        total_patients = len(set(train_set.patient_ids) | set(val_set.patient_ids))
        print(f">> Cross-validation fold {args.cv_index + 1}/{args.cv_folds}: {len(train_set)} train / {len(val_set)} val slices (patients: {len(train_set.patient_ids)} train / {len(val_set.patient_ids)} val, total={total_patients})")
    else:
        train_set = SliceDataset('train',
                                 root_dir,
                                 img_transform=img_transform,
                                 gt_transform=gt_transform,
                                 debug=getattr(args, "debug", False),
                                 augment=aug,
                                 context=args.context)
        val_set = SliceDataset('val',
                               root_dir,
                               img_transform=img_transform,
                               gt_transform=gt_transform,
                               debug=getattr(args, "debug", False),
                               augment=None,
                               context=args.context)

    loader_workers = 4 if use_crossval else 0

    train_loader = DataLoader(train_set,
                              batch_size=B,
                              num_workers=loader_workers,
                              shuffle=True)

    val_loader = DataLoader(val_set,
                            batch_size=B,
                            num_workers=loader_workers,
                            shuffle=False)

    # --- Scheduler ---
    total_steps = len(train_loader) * max(int(args.epochs), 1)
    if total_steps == 0:
        scheduler = None
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=lr * 10,
            total_steps=total_steps,
            pct_start=0.3,
            anneal_strategy='cos',
            div_factor=10,
            final_div_factor=100
        )

    args.dest.mkdir(parents=True, exist_ok=True)
    return (net, optimizer, scheduler, device, train_loader, val_loader, K)


def runTraining(args):
    print(f">>> Setting up to train on {args.dataset} with {args.mode} using {args.optimizer} optimizer")
    net, optimizer, scheduler, device, train_loader, val_loader, K = setup(args)

    loss_choice = args.loss_type[0] if isinstance(args.loss_type, (list, tuple)) else args.loss_type

    if loss_choice == 'CrossEntropy':
        if args.mode == "full":
            loss_fn = CrossEntropy(idk=list(range(K)))  # Supervise both background and foreground
        elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
            loss_fn = CrossEntropy(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
        else:
            raise ValueError(args.mode, args.dataset)
    elif loss_choice == 'CombinedLoss':
        if args.mode == "full":
            loss_fn = CombinedLoss(idk=list(range(K)))  # Supervise both background and foreground
        elif args.mode in ["partial"] and args.dataset == 'SEGTHOR':
            loss_fn = CombinedLoss(idk=[0, 1, 3, 4])  # Do not supervise the heart (class 2)
        else:
            raise ValueError(args.mode, args.dataset)
    else:
        raise ValueError(f"Unknown loss type: {loss_choice}")

    log_loss_tra: Tensor = torch.zeros((args.epochs, len(train_loader)))
    log_dice_tra: Tensor = torch.zeros((args.epochs, len(train_loader.dataset), K))
    log_loss_val: Tensor = torch.zeros((args.epochs, len(val_loader)))
    log_dice_val: Tensor = torch.zeros((args.epochs, len(val_loader.dataset), K))

    best_dice: float = 0.0
    best_epoch: int = -1

    for e in range(args.epochs):
        aug_ref = getattr(train_loader.dataset, "augment", None)
        if aug_ref is None:
            aug_ref = getattr(train_loader.dataset, "augmentation", None)
        if args.aug == 'online' and aug_ref is not None:
            if e < 8:
                aug_ref.cfg.p_roi_focus = 0.65
                aug_ref.cfg.rot_deg = 6.0
                aug_ref.cfg.shear_deg = 4.0
                aug_ref.cfg.translate = 0.008
                aug_ref.cfg.p_rot90 = 0.0
                aug_ref.cfg.p_elastic = 0.0
            else:
                aug_ref.cfg.p_roi_focus = 0.60
                aug_ref.cfg.rot_deg = 8.0
                aug_ref.cfg.shear_deg = 5.0
                aug_ref.cfg.translate = 0.010
                aug_ref.cfg.p_rot90 = 0.10
                aug_ref.cfg.p_elastic = 0.20
                aug_ref.cfg.elastic_sigma = 8.0
                aug_ref.cfg.elastic_alpha = 1.2

        for m in ['train', 'val']:
            if m == 'train':
                net.train()
                opt = optimizer
                sched = scheduler  # Use the scheduler
                cm = Dcm
                desc = f">> Training   ({e: 4d})"
                loader = train_loader
                log_loss = log_loss_tra
                log_dice = log_dice_tra
            elif m == 'val':
                net.eval()
                opt = None
                sched = None  # No scheduler update on validation
                cm = torch.no_grad
                desc = f">> Validation ({e: 4d})"
                loader = val_loader
                log_loss = log_loss_val
                log_dice = log_dice_val
            else:
                raise ValueError(f"Unknown phase {m}")

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

                    # ==== forward (compatible with main output + two auxiliary heads) ====
                    out = net(img)  # If it is ENet_enhance, out will be (logits, aux4, aux3)
                    if isinstance(out, tuple):
                        pred_logits, aux4, aux3 = out
                    else:
                        pred_logits, aux4, aux3 = out, None, None

                    pred_probs = F.softmax(1 * pred_logits, dim=1)  # 1 is the temperature parameter

                    # Metrics computation, not used for training
                    pred_seg = probs2one_hot(pred_probs)  # float {0,1}
                    pred_seg_bool = pred_seg.bool()  # -> bool for bitwise &
                    gt_bool = gt.bool()  # keep a bool copy for Dice
                    log_dice[e, j:j + B, :] = dice_coef(pred_seg_bool, gt_bool)  # One DSC value per sample and per class

                    loss = loss_fn(pred_probs, gt)
                    if aux4 is not None:
                        aux4_probs = F.softmax(aux4, dim=1)
                        loss = loss + 0.3 * loss_fn(aux4_probs, gt)

                    if aux3 is not None:
                        aux3_probs = F.softmax(aux3, dim=1)
                        loss = loss + 0.2 * loss_fn(aux3_probs, gt)
                    log_loss[e, i] = loss.item()  # One loss value per batch (averaged in the loss)

                    if opt:  # Only for training
                        loss.backward()
                        opt.step()
                        if sched is not None:
                            sched.step()  # Step the scheduler after the optimizer step

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

    csv_file = args.dest / "training_metrics.csv"

    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        # Header
        header = ["epoch", "train_loss", "val_loss", "train_dice", "val_dice"] + [f"train_dice_class{k}" for k in range(1, K)] + [f"val_dice_class{k}" for k in range(1, K)]
        writer.writerow(header)

        for e in range(args.epochs):
            train_loss_mean = log_loss_tra[e].mean().item()
            val_loss_mean = log_loss_val[e].mean().item()
            train_dice_mean = log_dice_tra[e, :, 1:].mean().item()  # exclude background
            val_dice_mean = log_dice_val[e, :, 1:].mean().item()
            # Dice per class
            train_dice_classes = [log_dice_tra[e, :, k].mean().item() for k in range(1, K)]
            val_dice_classes = [log_dice_val[e, :, k].mean().item() for k in range(1, K)]

            row = [e, train_loss_mean, val_loss_mean, train_dice_mean, val_dice_mean] + train_dice_classes + val_dice_classes
            writer.writerow(row)

        print(f">>> Metrics exported to {csv_file}")

        # I save it at each epochs, in case the code crashes or I decide to stop it early
        np.save(args.dest / "loss_tra.npy", log_loss_tra)
        np.save(args.dest / "dice_tra.npy", log_dice_tra)
        np.save(args.dest / "loss_val.npy", log_loss_val)
        np.save(args.dest / "dice_val.npy", log_dice_val)

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


def summarize_crossval_runs(base_dest: Path, summaries: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    if not summaries:
        return {}

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
                'std': float(np.std(values)) if len(values) > 1 else 0.0,
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
    return aggregated


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=12, type=int)
    parser.add_argument('--dataset', default='TOY2', choices=datasets_params.keys())
    parser.add_argument('--mode', default='full', choices=['partial', 'full'])
    parser.add_argument('--loss_type', nargs='+', default=['CrossEntropy'],
                        choices=['CrossEntropy', 'CombinedLoss'],
                        help="One or more loss types to try.")
    parser.add_argument('--dest', type=Path, required=True,
                        help="Destination directory to save results.")
    parser.add_argument('--optimizer', nargs='+', default=['adamw'],
                        choices=['adam', 'adamw', 'radam', 'sgd', 'lion'],
                        help="One or more optimizers to try.")
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--n_runs', default=1, type=int)
    parser.add_argument('--arch', nargs='+', default=['enet'],
                        choices=['enet', 'enetx', 'vit'],
                        help="One or more architectures to try.")
    parser.add_argument('--aug', nargs='+', default=['none'],
                        choices=['none', 'online'],
                        help="One or more augmentation modes to try.")

    parser.add_argument("--do_norm", action="store_true",)
    parser.add_argument("--norm_lo", type=float, default=0.5)
    parser.add_argument("--norm_hi", type=float, default=99.5)
    parser.add_argument("--do_median", action="store_true")
    parser.add_argument("--median_size", type=int, default=3)
    parser.add_argument("--do_clahe", action="store_true")
    parser.add_argument("--clahe_clip", type=float, default=4.0)
    parser.add_argument("--clahe_grid", type=int, default=8)
    parser.add_argument('--preproc', action='store_true')

    parser.add_argument('--context', type=int, default=1,
                        help="Context size for the 25D dataset.")

    parser.add_argument('--cv-folds', type=int, default=0,
                        help="Number of folds for cross-validation; set >1 to enable.")
    parser.add_argument('--cv-index', type=int, default=0,
                        help="Fold index to train when cross-validation is enabled.")
    parser.add_argument('--cv-run-all', action='store_true',
                        help="Train sequentially on every fold when cross-validation is enabled.")
    parser.add_argument('--cv-seed', type=int, default=42,
                        help="Random seed used to shuffle samples before fold splits.")

    args = parser.parse_args()

    if isinstance(args.loss_type, str):
        args.loss_type = [args.loss_type]
    if isinstance(args.optimizer, str):
        args.optimizer = [args.optimizer]
    if isinstance(args.arch, str):
        args.arch = [args.arch]
    if isinstance(args.aug, str):
        args.aug = [args.aug]

    orig_dest = args.dest
    pprint(args)

    if args.preproc:
        src_root = Path(f"data/{args.dataset}")
        preproc_root = Path(f"data/{args.dataset}_preproc")

        if preproc_root.exists():
            print(f"\n>>> Preprocessed dataset already exists: {preproc_root}")
            args.dataset = preproc_root.name
        else:
            print(f"\n>>> Running preprocessing on: {src_root}")
            out_root = run_preprocess_slices(
                src_root=src_root,
                do_norm=args.do_norm,
                norm_lo=args.norm_lo,
                norm_hi=args.norm_hi,
                do_median=args.do_median,
                median_size=args.median_size,
                do_clahe=args.do_clahe,
                clahe_clip=args.clahe_clip,
                clahe_grid=args.clahe_grid
            )

            # switch dataset
            if args.dataset.upper() == "SEGTHOR_CLEAN":
                args.dataset = out_root.name  # e.g. "SEGTHOR_CLEAN_preproc"
                print(f">>> Using preprocessed dataset: {args.dataset}")

    if args.cv_folds < 0:
        raise ValueError("--cv-folds must be >= 0")

    use_crossval = args.cv_folds and args.cv_folds > 1

    loss_types = args.loss_type
    optimizers = args.optimizer
    archs = args.arch
    augs = args.aug

    combos = list(product(loss_types, optimizers, archs, augs))
    print(f"\n>>> Running {len(combos)} combinations.")

    if use_crossval:
        if args.cv_run_all:
            folds_to_run = list(range(args.cv_folds))
        else:
            if not (0 <= args.cv_index < args.cv_folds):
                raise ValueError(f"--cv-index must be in [0, {args.cv_folds - 1}] when cross-validation is enabled")
            folds_to_run = [args.cv_index]

        cv_summary_rows = []
        for (loss_type, optimizer, arch, aug) in combos:
            combo_name = f"{loss_type}_{optimizer}_{arch}_{aug}"
            print(f"\n============================")
            print(f" Running combination: {combo_name}")
            print(f"============================")

            combo_dest = orig_dest / combo_name
            combo_dest.mkdir(parents=True, exist_ok=True)

            for run_idx in range(args.n_runs):
                print(f"\n--- Run {run_idx + 1}/{args.n_runs} for {combo_name} ---")
                run_dest = combo_dest / f"run_{run_idx + 1}"
                run_dest.mkdir(parents=True, exist_ok=True)

                fold_summaries: list[dict[str, Any]] = []
                for fold_idx in folds_to_run:
                    print(f"\n>>> Fold {fold_idx + 1}/{args.cv_folds}")
                    fold_args = argparse.Namespace(**vars(args))
                    fold_args.loss_type = loss_type
                    fold_args.optimizer = optimizer
                    fold_args.arch = arch
                    fold_args.aug = aug
                    fold_args.dest = run_dest / f"fold{fold_idx:02d}"
                    fold_args.cv_index = fold_idx
                    fold_args.cv_run_all = False
                    summary = runTraining(fold_args)
                    if summary:
                        fold_summaries.append(summary)

                aggregated = summarize_crossval_runs(run_dest, fold_summaries)
                if aggregated:
                    cv_summary_rows.append({
                        "combo_name": combo_name,
                        "run": run_idx + 1,
                        "best_dice_mean": aggregated.get('best_dice', {}).get('mean', float('nan')),
                        "best_dice_std": aggregated.get('best_dice', {}).get('std', float('nan')),
                        "final_val_dice_mean": aggregated.get('final_val_dice', {}).get('mean', float('nan')),
                        "final_val_dice_std": aggregated.get('final_val_dice', {}).get('std', float('nan')),
                        "final_val_loss_mean": aggregated.get('final_val_loss', {}).get('mean', float('nan')),
                        "final_val_loss_std": aggregated.get('final_val_loss', {}).get('std', float('nan')),
                    })
        if cv_summary_rows:
            summary_df = pd.DataFrame(cv_summary_rows)
            summary_path = orig_dest / "cv_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\n>>> Cross-validation summary saved to: {summary_path}")
            print(summary_df)
        return

    # Non cross-validation path ----------------------------------------
    summary_rows = []

    for (loss_type, optimizer, arch, aug) in combos:
        combo_name = f"{loss_type}_{optimizer}_{arch}_{aug}"
        print(f"\n============================")
        print(f" Running combination: {combo_name}")
        print(f"============================")

        combo_dest = orig_dest / combo_name
        combo_dest.mkdir(parents=True, exist_ok=True)

        run_metrics = []

        for run_idx in range(args.n_runs):
            print(f"\n--- Run {run_idx + 1}/{args.n_runs} for {combo_name} ---")

            run_dest = combo_dest / f"run_{run_idx + 1}"
            run_dest.mkdir(parents=True, exist_ok=True)

            run_args = argparse.Namespace(**vars(args))
            run_args.loss_type = loss_type
            run_args.optimizer = optimizer
            run_args.arch = arch
            run_args.aug = aug
            run_args.dest = run_dest

            runTraining(run_args)

            csv_path = run_dest / "training_metrics.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                last_epoch = df.iloc[-1]
                run_metrics.append({
                    "train_loss": last_epoch["train_loss"],
                    "val_loss": last_epoch["val_loss"],
                    "train_dice": last_epoch["train_dice"],
                    "val_dice": last_epoch["val_dice"],
                })
            else:
                print(f"Warning: Missing metrics file for {run_dest}")

        if run_metrics:
            avg_train_loss = np.mean([m["train_loss"] for m in run_metrics])
            avg_val_loss = np.mean([m["val_loss"] for m in run_metrics])
            avg_train_dice = np.mean([m["train_dice"] for m in run_metrics])
            avg_val_dice = np.mean([m["val_dice"] for m in run_metrics])

            summary_rows.append({
                "combo_name": combo_name,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss,
                "train_dice": avg_train_dice,
                "val_dice": avg_val_dice,
            })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = orig_dest / "summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\n>>> Summary saved to: {summary_path}")
        print(summary_df)


if __name__ == '__main__':
    main()
