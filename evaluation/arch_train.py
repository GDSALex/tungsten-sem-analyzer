"""Grouped cross-validation of network variants with the training routine of the final code.

One call trains one fold of one variant by calling tungsten_analyzer.train.train() of the final
code (github.com/GDSALex/tungsten-sem-analyzer, c1f25c9) unchanged.  Only three things are
substituted from outside, without touching the repository:

  * build_dataloaders returns the training side of the fold (all of it, with the strong
    augmentation of dataset.get_transforms) and, for the epoch log only, 16 of those images
    without augmentation; no image of the test fold is ever loaded during training;
  * build_unet returns the network of the variant (variants other than the U-Net of the code);
  * the schedule is fixed: 180 epochs, no early stopping, no periodic checkpoints, and the
    weights of the last epoch are kept, so no image takes part in a checkpoint selection.

Then the probability maps of the test images are written with the two passes of
inference.UNetPredictor (global pass at 512 px and tiled pass at the original resolution,
both with four-flip test time augmentation).

    python arch_train.py --variant unet_r34 --scheme edge5_r1 --fold 0
    python arch_train.py --variant unet_r34 --scheme all            # final model, 99 images
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402  (puts the final code on sys.path)

import albumentations as A  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

import segmentation_models_pytorch as smp  # noqa: E402
import tungsten_analyzer.train as T  # noqa: E402
from tungsten_analyzer.dataset import TungstenSEMDataset, _FilteredDataset, get_transforms  # noqa: E402
from tungsten_analyzer.inference import UNetPredictor  # noqa: E402
from tungsten_analyzer.model import build_unet_deep_supervised  # noqa: E402

RUNS = RC.REV / "arch_runs"
SUBMITTED_LOSS = dict(bce_weight=0.4, pos_weight=[8.0, 20.0], focal_gamma=2.0,
                      channel_weights=[1.0, 2.0])
PLAIN_LOSS = dict(bce_weight=0.5, pos_weight=[1.0, 1.0], focal_gamma=0.0, channel_weights=None)


def _smp(arch, encoder, **kw):
    return lambda: getattr(smp, arch)(encoder_name=encoder, encoder_weights="imagenet",
                                      in_channels=3, classes=2, activation=None, **kw)


# name: (network builder or None for build_unet of the code, deep supervision, loss, crops)
VARIANTS = {
    # recipe of the submitted model (checkpoint configuration): U-Net, ResNet34, scSE
    "unet_r34": dict(build=None, ds=False, loss=SUBMITTED_LOSS, crops="code"),
    # default of the final code: the same plus deep supervision and the crack-aware loss
    "unet_r34_ds": dict(build=None, ds=True, loss=SUBMITTED_LOSS, crops="code"),
    # plain pretrained U-Net, unweighted BCE + Dice (baseline requested by the reviewer)
    "unet_plain": dict(build=_smp("Unet", "resnet34"), ds=False, loss=PLAIN_LOSS, crops="code"),
    "unetpp_r34": dict(build=_smp("UnetPlusPlus", "resnet34", decoder_attention_type="scse"),
                       ds=False, loss=SUBMITTED_LOSS, crops="code"),
    "dlv3p_r34": dict(build=_smp("DeepLabV3Plus", "resnet34"), ds=False, loss=SUBMITTED_LOSS,
                      crops="code"),
    "unet_effb4": dict(build=_smp("Unet", "efficientnet-b4", decoder_attention_type="scse"),
                       ds=False, loss=SUBMITTED_LOSS, crops="code"),
    "unet_convnext": dict(build=_smp("Unet", "tu-convnext_tiny", decoder_attention_type="scse"),
                          ds=False, loss=SUBMITTED_LOSS, crops="code"),
    "segformer_b2": dict(build=_smp("Segformer", "mit_b2"), ds=False, loss=SUBMITTED_LOSS,
                         crops="code"),
    # the submitted recipe trained on crops close to the original resolution (tiled inference)
    "unet_r34_native": dict(build=None, ds=False, loss=SUBMITTED_LOSS, crops="native"),
}


def native_transforms(img_size: int = 512) -> A.Compose:
    """get_transforms('train') of the final code with its first step replaced: crops of 25 to
    50 % of the area (425 to 600 px for a 1024 x 707 region) resized to 512 px, i.e. at 0.85 to
    1.2 times the original resolution instead of about 0.5 to 0.6 times."""
    base = get_transforms("train", img_size)
    steps = list(base.transforms)
    assert isinstance(steps[0], A.RandomResizedCrop), steps[0]
    steps[0] = A.RandomResizedCrop(size=(img_size, img_size), scale=(0.25, 0.5),
                                   ratio=(0.85, 1.18), p=1.0)
    return A.Compose(steps)


def build_network(name: str) -> torch.nn.Module:
    v = VARIANTS[name]
    if v["ds"]:
        return build_unet_deep_supervised("resnet34", "imagenet", 2)
    if v["build"] is None:
        return T.build_unet(encoder_name="resnet34", encoder_weights="imagenet", num_classes=2)
    return v["build"]()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader(names, transform, shuffle, drop_last, workers):
    base = TungstenSEMDataset(image_dir=str(RC.IMG_DIR), transform=transform,
                              strip_fraction=RC.STRIP_FRACTION)
    ds = _FilteredDataset(base, allowed_names={f"{n}.jpg" for n in names})
    assert len(ds) == len(names), (len(ds), len(names))
    return DataLoader(ds, batch_size=4, shuffle=shuffle, num_workers=workers, pin_memory=True,
                      drop_last=drop_last, persistent_workers=workers > 0,
                      worker_init_fn=RC.worker_init if workers > 0 else None)


@torch.no_grad()
def write_maps(model, names, img_dir: Path, out_dir: Path, device) -> None:
    """Global pass and tiled pass of inference.UNetPredictor, both with four-flip TTA."""
    pred = UNetPredictor.__new__(UNetPredictor)
    pred.model, pred.device, pred.img_size = model.eval(), device, 512
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {p.stem: p for p in RC.list_images(img_dir)}
    for n in names:
        gray = RC.roi_of(RC.read_gray(paths[n]))
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        H, W = gray.shape
        glob = pred._predict_single(rgb, H, W)
        tile = pred._predict_tiled(rgb, H, W)
        np.savez_compressed(out_dir / f"{n}.npz", glob=glob.astype(np.float16),
                            tile=(tile if tile is not None else np.zeros_like(glob)).astype(np.float16),
                            has_tile=tile is not None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=sorted(VARIANTS))
    ap.add_argument("--scheme", required=True, choices=["edge5_r1", "edge5_r2", "loso", "all"])
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=180)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="3 epochs, encoder frozen for 1, maps for 2 test images (checks only)")
    a = ap.parse_args()
    if a.smoke:
        a.epochs = 3

    RC.set_low_priority_off()
    man = RC.load_manifest()
    if a.scheme == "all":
        train_names, test = sorted(man["name"]), []
        seed = 4000
    else:
        col = f"fold_{a.scheme}"
        test = sorted(man.loc[man[col] == a.fold, "name"])
        train_names = sorted(man.loc[man[col] != a.fold, "name"])
        seed = 1000 + {"edge5_r1": 0, "edge5_r2": 100, "loso": 200}[a.scheme] + a.fold
    tag = "all" if a.scheme == "all" else f"{a.scheme}_f{a.fold}"
    out = Path(a.out) if a.out else RUNS / a.variant / tag
    out.mkdir(parents=True, exist_ok=True)
    if (out / "done.json").exists():
        print("already done:", out)
        return
    v = VARIANTS[a.variant]
    info = {"variant": a.variant, "scheme": a.scheme, "fold": a.fold, "seed": seed,
            "epochs": a.epochs, "deep_supervision": v["ds"], "loss": v["loss"],
            "crops": v["crops"], "test": test, "train": train_names,
            "code": "github.com/GDSALex/tungsten-sem-analyzer c1f25c9, train.train()",
            "torch": torch.__version__, "smp": smp.__version__}
    (out / "fold_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    print(f"[{a.variant} {tag}] train {len(train_names)}  test {len(test)}  seed {seed}", flush=True)

    seed_everything(seed)
    torch.backends.cudnn.benchmark = True
    tr = native_transforms() if v["crops"] == "native" else get_transforms("train", 512)
    rng = random.Random(seed)
    log_names = sorted(rng.sample(train_names, 16))
    tl = loader(train_names, tr, True, True, a.workers)
    ll = loader(log_names, get_transforms("val", 512), False, False, 0)

    # substitutions in the namespace of the training module of the final code
    T.build_dataloaders = lambda **kw: (tl, ll)
    if v["build"] is not None:
        T.build_unet = lambda **kw: v["build"]()
    cfg = dict(T.DEFAULT_CFG)
    cfg.update({"img_size": 512, "batch_size": 4, "total_epochs": a.epochs,
                "use_deep_supervision": v["ds"], "patience": 10 ** 9, "save_every": 10 ** 9,
                "checkpoint_dir": str(out), "num_workers": a.workers, **v["loss"]})
    if a.smoke:
        cfg.update({"freeze_epochs": 1, "warmup_epochs": 1})
        test = test[:2]
    t0 = time.time()
    T.train(cfg)
    minutes = (time.time() - t0) / 60
    peak_train = torch.cuda.max_memory_reserved() / 2 ** 30
    (out / "best_model.pth").unlink(missing_ok=True)   # selected on training images: not used

    device = torch.device("cuda")
    model = build_network(a.variant).to(device)
    state = torch.load(out / "last_model.pth", map_location=device, weights_only=False)["model_state"]
    model.load_state_dict(state)
    t1 = time.time()
    if test:
        write_maps(model, test, RC.IMG_DIR, out / "probs_test", device)
    json.dump({"epochs_run": a.epochs, "minutes": round(minutes, 2),
               "map_minutes": round((time.time() - t1) / 60, 2),
               "peak_reserved_GB": round(peak_train, 2),
               "params_M": round(sum(p.numel() for p in model.parameters()) / 1e6, 2)},
              open(out / "done.json", "w"), indent=1)
    print(f"done {out}  {minutes:.1f} min", flush=True)


if __name__ == "__main__":
    main()
