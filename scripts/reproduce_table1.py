"""Reproduce Table 1 of the paper.

REVISION 1 NOTE.  This script reproduces Table 1 of the first version of the
manuscript, which is superseded.  Those figures are in-sample (the network had been
trained or selected on every image), and the crack figure is the legacy score of that
version, which is not an IoU.  With the default thresholds (0.5 / 0.30) the script
returns the printed values; the text of the first version stated a crack threshold of
0.25 (use --thr-crack 0.25).  The out-of-fold evaluation of the revised manuscript
(edge-grouped cross-validation, strict crater IoU and length-weighted crack
centreline scores) is produced by the evaluation code archived with the revision.

The script runs both methods over the annotated dataset, scores them with the single
protocol of :mod:`tungsten_analyzer.metrics`, and prints the medians and means for
the whole annotated set and for the training and held-out subsets separately.

The classical pipeline is run with the parameters found by the evolutionary search,
which are the ones hard-coded in ``classical_pipeline.CFG``, and the network is run
with a single global forward pass at 512 pixels and four-fold flip test-time
augmentation. Neither method is given any advantage in how it is measured.

The train and held-out subsets are recovered by replaying the split of the training
run, a plain random shuffle of the sorted file names under seed 42 with the first
15 per cent held out. Keep that seed unless you have retrained the model, or the
subsets will no longer correspond to what the checkpoint actually saw.

Usage::

    python scripts/reproduce_table1.py --data-dir ./data --checkpoint checkpoints/best_model.pth

Add ``--skip-classical`` to evaluate only the network, which is considerably faster.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from tungsten_analyzer import metrics as M
from tungsten_analyzer.model import build_unet, build_unet_deep_supervised
from tungsten_analyzer.validation_figures import predict_image

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def load_network(checkpoint_path: str, device: torch.device):
    """Restore the trained network, handling both the plain and deep-supervised saves."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg = ckpt.get("cfg", {})
    deep_supervised = any(k.startswith("base.") for k in ckpt["model_state"])
    build = build_unet_deep_supervised if deep_supervised else build_unet
    try:
        model = build(encoder_name=cfg.get("encoder", "resnet34"),
                      encoder_weights=None,
                      num_classes=cfg.get("num_classes", 2)).to(device)
    except TypeError:                      # build_unet_deep_supervised takes no weights arg
        model = build(encoder_name=cfg.get("encoder", "resnet34"),
                      num_classes=cfg.get("num_classes", 2)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Checkpoint from epoch {ckpt.get('epoch', '?')} loaded on {device}")
    return model


def validation_split(image_names, val_fraction: float = 0.15, seed: int = 42) -> set:
    """Replay the split used during training and return the held-out file names."""
    shuffled = list(image_names)
    random.seed(seed)
    random.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * val_fraction))
    return set(shuffled[:n_val])


def classical_prediction(roi_gray: np.ndarray, um_per_px: float):
    """Run the tuned classical pipeline and return its crater and crack masks."""
    from tungsten_analyzer import classical_pipeline as cp

    edge_end, artifact_start = cp.compute_zones_dynamic(roi_gray, bottom_pct=0.02)
    empty = np.zeros_like(roi_gray, dtype=bool)
    raw_craters, dark_labels = cp.detect_craters(roi_gray, empty, edge_end, artifact_start)
    valid, _ = cp.filter_artifacts(raw_craters, artifact_start)
    _, crater_details = cp.classify_craters(valid, um_per_px)
    labels = [c["label"] for c in crater_details]
    crater_mask = np.isin(dark_labels, labels) if labels else np.zeros_like(roi_gray, dtype=bool)

    min_skel_px = max(10, int(15.0 / um_per_px))
    crack_mask, _, _ = cp.detect_cracks(
        roi_gray, crater_mask, edge_end, artifact_start,
        frangi_thresh=0.18, min_skel_px=min_skel_px,
        min_area_px=max(20, min_skel_px // 2))
    return np.asarray(crater_mask, bool), np.asarray(crack_mask, bool)


def print_block(title: str, records, counts=None) -> None:
    stats = M.summarise(records)
    print(f"\n  {title}")
    print(f"    craters n = {stats['n_crater']:<4d} cracks n = {stats['n_crack']}")
    print("    median   crater IoU {crater_iou_median:.3f}  F1 {crater_f1_median:.3f}"
          "   crack IoU {crack_iou_median:.3f}  F1 {crack_f1_median:.3f}".format(**stats))
    print("    mean     crater IoU {crater_iou_mean:.3f}  F1 {crater_f1_mean:.3f}"
          "   crack IoU {crack_iou_mean:.3f}  F1 {crack_f1_mean:.3f}".format(**stats))
    if counts:
        print(f"    detected on {counts['crack_detected']}/{counts['crack_annotated']} "
              f"annotated crack images, fired on {counts['crack_false_alarms']}/"
              f"{counts['crack_clean']} clean ones")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--data-dir", default="./data",
                        help="root directory holding images/ and masks/")
    parser.add_argument("--checkpoint", default="checkpoints/best_model.pth")
    parser.add_argument("--img-size", type=int, default=512,
                        help="input size of the global forward pass")
    parser.add_argument("--strip-fraction", type=float, default=M.METADATA_STRIP_FRACTION,
                        help="fraction of the frame height occupied by the SEM metadata bar")
    parser.add_argument("--thr-crater", type=float, default=0.5,
                        help="threshold of the crater channel (default 0.5)")
    parser.add_argument("--thr-crack", type=float, default=0.30,
                        help="threshold of the crack channel.  The default 0.30 reproduces the "
                             "figures printed in the first version; the text of that version "
                             "stated 0.25")
    parser.add_argument("--skip-classical", action="store_true",
                        help="evaluate only the network")
    parser.add_argument("--save", default=None,
                        help="optional path for the per-image records, as JSON")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_network(args.checkpoint, device)

    data_dir = Path(args.data_dir)
    image_dir, mask_dir = data_dir / "images", data_dir / "masks"
    names = sorted(p.name for p in image_dir.iterdir()
                   if p.suffix.lower() in IMAGE_EXTENSIONS)
    held_out = validation_split(names)
    print(f"  {len(names)} images, {len(held_out)} of them held out from training")

    unet_records, classical_records = [], []
    for name in names:
        json_path = mask_dir / (Path(name).stem + ".json")
        if not json_path.exists():
            continue

        image = cv2.imread(str(image_dir / name))
        full_h, full_w = image.shape[:2]
        roi = cv2.cvtColor(image[:full_h - int(full_h * args.strip_fraction), :],
                           cv2.COLOR_BGR2GRAY)
        h, w = roi.shape

        gt_crater, gt_crack = M.load_annotation(json_path, h, w,
                                                full_height=full_h, full_width=full_w)

        pred_crater, pred_crack, _, _ = predict_image(model, roi, device, args.img_size,
                                                      threshold_crater=args.thr_crater,
                                                      threshold_crack=args.thr_crack)
        record = M.score_image(gt_crater, gt_crack, pred_crater, pred_crack)
        record["name"] = Path(name).stem
        record["held_out"] = name in held_out
        unet_records.append(record)

        if not args.skip_classical:
            from tungsten_analyzer.measurement import read_scale
            um = read_scale(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))["um_per_px"]
            if um is None:      # no default pixel size any more (revision 1)
                print(f"  {name}: pixel size not verified, classical pipeline skipped")
                continue
            c_crater, c_crack = classical_prediction(roi, um)
            rec = M.score_image(gt_crater, gt_crack, c_crater, c_crack)
            rec["name"] = Path(name).stem
            rec["held_out"] = name in held_out
            classical_records.append(rec)

    print("\n" + "=" * 78)
    print("  TABLE 1")
    print("=" * 78)

    if classical_records:
        print_block("Classical pipeline, all images",
                    classical_records, M.image_level_counts(classical_records))

    print_block("U-Net, all images", unet_records, M.image_level_counts(unet_records))
    print_block("U-Net, training subset", [r for r in unet_records if not r["held_out"]])
    print_block("U-Net, held-out subset", [r for r in unet_records if r["held_out"]])

    if args.save:
        payload = {"unet": unet_records, "classical": classical_records}
        Path(args.save).write_text(json.dumps(payload, indent=1), encoding="utf-8")
        print(f"\n  Per-image records written to {args.save}")


if __name__ == "__main__":
    main()
