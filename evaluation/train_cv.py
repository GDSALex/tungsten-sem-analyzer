"""Grouped cross-validation of the U-Net recipe (reviewer comments 1, 8 and 9).

One call trains one outer fold of one configuration and then writes the probability maps
of the images it never saw.  Model and checkpoint selection use only the training side:
an inner validation subset, grouped by exposure position, plays the role that the 14
random validation images played in the submitted manuscript.

Configurations
    A  the recipe of the submitted model, reconstructed from the saved checkpoint:
       smp U-Net, ResNet34 ImageNet encoder, scSE decoder attention, 0.4 x focal-modulated
       BCE (pos_weight 8/20, gamma 2) + 0.6 x channel-weighted soft Dice (1/2)
    B  plain baseline: the same U-Net without scSE, unweighted BCE + Dice (0.5/0.5)
    C  A without scSE attention
    D  A plus deep supervision and the two crack-specific terms of model.CrackAwareLoss
       (boundary-weighted BCE 0.3, interior-miss 0.1, auxiliary weights 1.0/0.4/0.2/0.2)

Every other setting is the one stored in the checkpoint of the submitted model: 512 x 512
input, batch 4, no gradient accumulation, 15 epochs with a frozen encoder (BatchNorm
statistics still updated) followed by differential learning rates, AdamW, cosine
schedules, mixed precision, gradient clipping at 1, at most 180 epochs, early stopping
after 30 epochs without improvement of 0.5 mean IoU + 0.5 mean F1 (micro-averaged,
thresholds 0.5 and 0.25) on the inner validation subset.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402  (sets sys.dont_write_bytecode, repo on sys.path)

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

import segmentation_models_pytorch as smp
from tungsten_analyzer.dataset import TungstenSEMDataset, get_transforms, _FilteredDataset
from tungsten_analyzer.model import (CombinedLoss, CrackAwareLoss, MetricsAccumulator,
                                     build_unet, build_unet_deep_supervised,
                                     freeze_encoder, unfreeze_encoder, get_param_groups)
from tungsten_analyzer.inference import UNetPredictor

BASE_CFG = {
    "img_size": 512, "batch_size": 4, "freeze_epochs": 15, "total_epochs": 180,
    "lr_decoder": 1e-3, "lr_encoder": 1e-4, "weight_decay": 1e-4, "warmup_epochs": 3,
    "patience": 30, "threshold": 0.5, "threshold_crack": 0.25,
}

CONFIGS = {
    "A": dict(scse=True, loss=dict(bce_weight=0.4, pos_weight=[8.0, 20.0], focal_gamma=2.0,
                                    channel_weights=[1.0, 2.0]), deep=False),
    "B": dict(scse=False, loss=dict(bce_weight=0.5, pos_weight=1.0, focal_gamma=0.0,
                                     channel_weights=None), deep=False),
    "C": dict(scse=False, loss=dict(bce_weight=0.4, pos_weight=[8.0, 20.0], focal_gamma=2.0,
                                     channel_weights=[1.0, 2.0]), deep=False),
    "D": dict(scse=True, loss=dict(bce_weight=0.4, pos_weight=[8.0, 20.0], focal_gamma=2.0,
                                    channel_weights=[1.0, 2.0]), deep=True),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(cfg_name: str) -> torch.nn.Module:
    c = CONFIGS[cfg_name]
    if c["deep"]:
        return build_unet_deep_supervised("resnet34", "imagenet", 2)
    if c["scse"]:
        return build_unet("resnet34", "imagenet", 2)
    return smp.Unet(encoder_name="resnet34", encoder_weights="imagenet", in_channels=3,
                    classes=2, activation=None, decoder_attention_type=None)


def inner_split(train_names: list[str], man, seed: int, frac: float = 0.15):
    """Hold out whole exposure positions until ~15 % of the training side is reached."""
    rng = random.Random(seed)
    sub = man.set_index("name").loc[train_names]
    positions = sorted(sub["position"].unique())
    rng.shuffle(positions)
    val: list[str] = []
    for pos in positions:
        enough = (len(val) >= frac * len(train_names)
                  and sub.loc[val, "has_crack"].sum() >= 3
                  and sub.loc[val, "has_crater"].sum() >= 3) if val else False
        if enough:
            break
        val += sorted(sub.index[sub["position"] == pos])
    tr = sorted(set(train_names) - set(val))
    return tr, sorted(val)


def make_loader(names, transform, shuffle, workers, batch_size, drop_last):
    base = TungstenSEMDataset(image_dir=str(RC.IMG_DIR), transform=transform,
                              strip_fraction=RC.STRIP_FRACTION)
    ds = _FilteredDataset(base, allowed_names={f"{n}.jpg" for n in names})
    assert len(ds) == len(names), (len(ds), len(names))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=workers,
                      pin_memory=True, drop_last=drop_last,
                      persistent_workers=workers > 0,
                      worker_init_fn=RC.worker_init if workers > 0 else None)


def run_epoch_train(model, loader, optimizer, criterion, device, scaler):
    model.train()
    tot, n = 0.0, 0
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            loss = criterion(out, y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        tot += loss.item()
        n += 1
    return tot / max(n, 1)


@torch.no_grad()
def run_epoch_eval(model, loader, criterion, common, device, thr, thr_k, metrics=True):
    model.eval()
    tot = tot_c = 0.0
    n = 0
    acc = MetricsAccumulator(class_names=["crater", "crack"])
    for batch in loader:
        x = batch["image"].to(device, non_blocking=True)
        y = batch["mask"].to(device, non_blocking=True)
        logits = model(x)
        tot += criterion(logits, y).item()
        tot_c += common(logits, y).item()
        n += 1
        if metrics:
            p = torch.sigmoid(logits)
            pred = torch.zeros_like(p)
            pred[:, 0] = (p[:, 0] > thr).float()
            pred[:, 1] = (p[:, 1] > thr_k).float()
            acc.update(pred, y)
    return tot / max(n, 1), tot_c / max(n, 1), (acc.compute() if metrics else {})


@torch.no_grad()
def write_probability_maps(model, names, out_dir: Path, device):
    """Global 512-px pass and tiled pass, both with 4-flip TTA, as in inference.py."""
    pred = UNetPredictor.__new__(UNetPredictor)
    pred.model, pred.device, pred.img_size = model.eval(), device, 512
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in names:
        gray = RC.roi_of(RC.read_gray(RC.IMG_DIR / f"{n}.jpg"))
        import cv2
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        H, W = gray.shape
        glob = pred._predict_single(rgb, H, W)
        tile = pred._predict_tiled(rgb, H, W)
        np.savez_compressed(out_dir / f"{n}.npz", glob=glob.astype(np.float16),
                            tile=(tile if tile is not None else np.zeros_like(glob)).astype(np.float16),
                            has_tile=tile is not None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, choices=sorted(CONFIGS))
    ap.add_argument("--scheme", required=True, choices=["edge5_r1", "edge5_r2", "loso"])
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=None, help="override, for smoke tests")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    RC.set_low_priority_off()
    man = RC.load_manifest()
    col = f"fold_{args.scheme}"
    test = sorted(man.loc[man[col] == args.fold, "name"])
    train_side = sorted(man.loc[man[col] != args.fold, "name"])
    rep_offset = {"edge5_r1": 0, "edge5_r2": 100, "loso": 200}[args.scheme]
    seed = args.seed if args.seed is not None else 1000 + rep_offset + args.fold
    inner_tr, inner_val = inner_split(train_side, man, seed)

    out = Path(args.out) if args.out else RC.CV_RUNS / f"{args.config}_{args.scheme}_f{args.fold}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        print("already done:", out)
        return

    cfg = dict(BASE_CFG)
    if args.epochs:
        cfg["total_epochs"] = args.epochs
    cfgx = CONFIGS[args.config]
    info = {"config": args.config, "scheme": args.scheme, "fold": args.fold, "seed": seed,
            "cfg": cfg, "model": {k: v for k, v in cfgx.items()},
            "test": test, "inner_train": inner_tr, "inner_val": inner_val,
            "torch": torch.__version__, "smp": smp.__version__}
    (out / "fold_info.json").write_text(json.dumps(info, indent=1))
    print(f"[{args.config} {args.scheme} f{args.fold}] train {len(inner_tr)}  inner-val "
          f"{len(inner_val)}  test {len(test)}  seed {seed}", flush=True)

    seed_everything(seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda")

    tl = make_loader(inner_tr, get_transforms("train", cfg["img_size"]), True, args.workers,
                     cfg["batch_size"], True)
    vl = make_loader(inner_val, get_transforms("val", cfg["img_size"]), False, 0,
                     cfg["batch_size"], False)
    tl_eval = make_loader(inner_tr, get_transforms("val", cfg["img_size"]), False, 0,
                          cfg["batch_size"], False)

    model = build_model(args.config).to(device)
    base_loss = CombinedLoss(**cfgx["loss"]).to(device)
    criterion = (CrackAwareLoss(base_loss, boundary_weight=0.3, continuity_weight=0.1,
                                deep_sup_weights=[1.0, 0.4, 0.2]).to(device)
                 if cfgx["deep"] else base_loss)
    # common loss used for the train/validation gap of every configuration (config A loss,
    # main output only, no augmentation, eval mode)
    common = CombinedLoss(**CONFIGS["A"]["loss"]).to(device)

    freeze_encoder(model)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=cfg["lr_decoder"], weight_decay=cfg["weight_decay"])
    scheduler = SequentialLR(optimizer, schedulers=[
        LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=cfg["warmup_epochs"]),
        CosineAnnealingLR(optimizer, T_max=cfg["total_epochs"] - cfg["warmup_epochs"], eta_min=1e-6)],
        milestones=[cfg["warmup_epochs"]])
    scaler = torch.cuda.amp.GradScaler()

    best, best_epoch, wait, history = -1.0, 0, 0, []
    t_start = time.time()
    for epoch in range(1, cfg["total_epochs"] + 1):
        t0 = time.time()
        if epoch == cfg["freeze_epochs"] + 1:
            unfreeze_encoder(model)
            optimizer = optim.AdamW(get_param_groups(model, base_lr=cfg["lr_decoder"]),
                                    weight_decay=cfg["weight_decay"])
            scheduler = CosineAnnealingLR(optimizer, T_max=cfg["total_epochs"] - cfg["freeze_epochs"],
                                          eta_min=1e-6)
        tr_loss = run_epoch_train(model, tl, optimizer, criterion, device, scaler)
        v_loss, v_common, vm = run_epoch_eval(model, vl, criterion, common, device,
                                              cfg["threshold"], cfg["threshold_crack"])
        scheduler.step()
        rec = {"epoch": epoch, "train_loss": tr_loss, "val_loss": v_loss,
               "val_common_loss": v_common, **vm, "sec": time.time() - t0}
        if epoch % 5 == 0 or epoch == 1:
            _, tr_common, _ = run_epoch_eval(model, tl_eval, criterion, common, device,
                                             cfg["threshold"], cfg["threshold_crack"], metrics=False)
            rec["train_common_loss_noaug"] = tr_common
        score = 0.5 * vm["mean_iou"] + 0.5 * vm["mean_f1"]
        rec["score"] = score
        history.append(rec)
        if score > best:
            best, best_epoch, wait = score, epoch, 0
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "score": score,
                        "config": args.config, "cfg": cfg}, out / "best_model.pth")
        else:
            wait += 1
        print(f"  ep {epoch:3d} loss {tr_loss:.4f}/{v_loss:.4f} common {v_common:.4f} "
              f"IoU c {vm['crater_iou']:.3f} k {vm['crack_iou']:.3f} score {score:.4f} "
              f"best {best:.4f}@{best_epoch} {rec['sec']:.1f}s", flush=True)
        (out / "history.json").write_text(json.dumps(history, indent=0))
        if wait >= cfg["patience"]:
            break

    # probability maps of the outer test images and of the inner validation images
    ck = torch.load(out / "best_model.pth", map_location=device)
    model.load_state_dict(ck["model_state"])
    model.eval()
    write_probability_maps(model, test, out / "probs_test", device)
    write_probability_maps(model, inner_val, out / "probs_innerval", device)
    (out / "done.json").write_text(json.dumps({
        "best_epoch": best_epoch, "best_score": best, "epochs_run": len(history),
        "minutes": (time.time() - t_start) / 60}, indent=1))
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
