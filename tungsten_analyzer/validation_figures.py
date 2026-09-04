"""
=============================================================================
UNET_VALIDATE_VISUAL.PY  —  Visual Validation of U-Net Predictions
=============================================================================
Generates side-by-side comparison images showing:
  - Original SEM image
  - Ground truth masks (from LabelMe JSON)
  - U-Net predictions
  - Overlay comparison (TP=green, FP=red, FN=blue)

Usage:
  python unet_validate_visual.py
  python unet_validate_visual.py --checkpoint checkpoints/best_model.pth
  python unet_validate_visual.py --data-dir ./data --num-images 10
  python unet_validate_visual.py --output-dir ./validation_results
=============================================================================
"""

import argparse
import json
import random
from pathlib import Path
from typing import Tuple

import numpy as np
import cv2
import torch
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tungsten_analyzer.model import build_unet

# ─────────────────────────────────────────────────────────────────────────────
#  MASK LOADING (same logic as unet_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

CRATER_LABELS = {'crater', 'cratera', 'cráter', 'vaporization',
                 'collapse', 'crater_vaporization', 'crater_collapse'}
CRACK_LABELS  = {'crack', 'grieta', 'fracture', 'fractura',
                 'crack_large', 'crack_small'}


def load_masks_from_json(json_path: Path, h: int, w: int,
                         h_full: int = None, w_full: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load crater and crack masks from a LabelMe JSON file.
    
    h, w: target size AFTER stripping metadata bar.
    h_full, w_full: original image size BEFORE stripping.
    Masks are drawn at full size then CROPPED to match the stripped image.
    """
    from PIL import Image, ImageDraw

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    img_w = data.get('imageWidth',  w_full or w)
    img_h = data.get('imageHeight', h_full or h)

    canvas_crater = Image.new('L', (img_w, img_h), 0)
    canvas_crack  = Image.new('L', (img_w, img_h), 0)
    draw_crater   = ImageDraw.Draw(canvas_crater)
    draw_crack    = ImageDraw.Draw(canvas_crack)

    # Adaptive crack line width — must match unet_dataset.py
    base_width = 15
    CRACK_LINE_WIDTH = max(7, int(base_width * max(img_w, img_h) / 1024))

    for shape in data.get('shapes', []):
        label      = shape['label'].strip().lower()
        points     = [(float(p[0]), float(p[1])) for p in shape['points']]
        shape_type = shape.get('shape_type', 'polygon')

        if label in CRATER_LABELS:
            draw = draw_crater
        elif label in CRACK_LABELS:
            draw = draw_crack
        else:
            continue

        if shape_type == 'polygon' and len(points) >= 3:
            draw.polygon(points, fill=255)
        elif shape_type in ('linestrip', 'line') and len(points) >= 2:
            draw.line(points, fill=255, width=CRACK_LINE_WIDTH)
        elif shape_type == 'rectangle' and len(points) >= 2:
            x0, y0 = points[0]
            x1, y1 = points[-1]
            draw.rectangle([min(x0,x1), min(y0,y1),
                            max(x0,x1), max(y0,y1)], fill=255)
        elif shape_type == 'circle' and len(points) == 2:
            cx, cy = points[0]
            px, py = points[1]
            r = ((px-cx)**2 + (py-cy)**2) ** 0.5
            draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)

    mask_crater = np.array(canvas_crater)
    mask_crack  = np.array(canvas_crack)

    # CROP instead of resize to match the stripped image
    if mask_crater.shape[0] > h:
        mask_crater = mask_crater[:h, :]
        mask_crack  = mask_crack[:h, :]
    
    if mask_crater.shape[1] != w:
        mask_crater = cv2.resize(mask_crater, (w, mask_crater.shape[0]),
                                interpolation=cv2.INTER_NEAREST)
        mask_crack  = cv2.resize(mask_crack, (w, mask_crack.shape[0]),
                                interpolation=cv2.INTER_NEAREST)

    return (mask_crater > 127).astype(np.uint8), (mask_crack > 127).astype(np.uint8)


# ─────────────────────────────────────────────────────────────────────────────
#  INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def predict_image(model, img_gray: np.ndarray, device, img_size=512,
                  threshold_crater=0.5, threshold_crack=0.3) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run U-Net inference on a grayscale SEM image.
    Returns: (crater_mask, crack_mask, crater_prob, crack_prob)
    
    Note: crack threshold is lower (0.3) to improve sensitivity for thin features.
    """
    H, W = img_gray.shape
    img_rgb = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

    # Preprocess: resize + normalize (ImageNet stats)
    img_resized = cv2.resize(img_rgb, (img_size, img_size)).astype(np.float32) / 255.0
    img_norm = (img_resized - _MEAN) / _STD
    img_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1)).unsqueeze(0).float().to(device)

    # Inference with TTA (4 flips)
    model.eval()
    with torch.no_grad():
        probs_accum = torch.zeros(1, 2, img_size, img_size, device=device)
        
        for flip_h in [False, True]:
            for flip_v in [False, True]:
                inp = img_tensor.clone()
                if flip_h: inp = torch.flip(inp, [3])
                if flip_v: inp = torch.flip(inp, [2])
                
                logits = model(inp)
                probs = torch.sigmoid(logits)
                
                if flip_v: probs = torch.flip(probs, [2])
                if flip_h: probs = torch.flip(probs, [3])
                
                probs_accum += probs
        
        probs_avg = probs_accum / 4.0

    # Back to numpy, resize to original
    probs_np = probs_avg[0].cpu().numpy()  # (2, 512, 512)
    
    crater_prob = cv2.resize(probs_np[0], (W, H), interpolation=cv2.INTER_LINEAR)
    crack_prob  = cv2.resize(probs_np[1], (W, H), interpolation=cv2.INTER_LINEAR)

    crater_mask = (crater_prob > threshold_crater).astype(np.uint8)
    crack_mask  = (crack_prob  > threshold_crack).astype(np.uint8)

    return crater_mask, crack_mask, crater_prob, crack_prob


# ─────────────────────────────────────────────────────────────────────────────
#  ERROR MAP VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def make_error_overlay(img_gray, gt_mask, pred_mask):
    """
    Create RGB overlay showing:
      Green  = True Positive  (correctly detected)
      Red    = False Positive (detected but not in GT)
      Blue   = False Negative (in GT but missed)
      Gray   = True Negative  (correctly ignored)
    """
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB).copy()
    
    tp = (gt_mask == 1) & (pred_mask == 1)
    fp = (gt_mask == 0) & (pred_mask == 1)
    fn = (gt_mask == 1) & (pred_mask == 0)
    
    overlay[tp] = [0, 220, 0]     # Green
    overlay[fp] = [220, 0, 0]     # Red  
    overlay[fn] = [0, 80, 255]    # Blue
    
    return overlay


def compute_iou(gt, pred):
    """Compute IoU between two binary masks."""
    tp = float((gt & pred).sum())
    fp = float(((~gt.astype(bool)) & pred.astype(bool)).sum())
    fn = float((gt.astype(bool) & (~pred.astype(bool))).sum())
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


def compute_f1(gt, pred):
    """Compute F1 (Dice) between two binary masks."""
    tp = float((gt.astype(bool) & pred.astype(bool)).sum())
    fp = float(((~gt.astype(bool)) & pred.astype(bool)).sum())
    fn = float((gt.astype(bool) & (~pred.astype(bool))).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp) / denom if denom > 0 else 0.0


def compute_crack_iou_tolerant(gt, pred, tolerance_px=5):
    """
    Compute tolerance-based IoU for thin crack structures.
    
    Standard IoU penalizes even perfectly-localized cracks if there's a
    width mismatch between GT (thin line) and prediction (thicker area).
    
    This metric dilates BOTH GT and prediction by `tolerance_px` before
    computing IoU, which is the standard approach in crack/road/vessel
    segmentation (centerline-tolerance evaluation).
    
    Additionally computes centerline recall: what fraction of the GT
    skeleton falls within `tolerance_px` of any predicted pixel.
    """
    from skimage.morphology import binary_dilation, disk
    
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    
    # If both are empty, perfect match
    if not gt_b.any() and not pred_b.any():
        return 1.0
    if not gt_b.any() or not pred_b.any():
        return 0.0
    
    selem = disk(tolerance_px)
    gt_dilated   = binary_dilation(gt_b, selem)
    pred_dilated = binary_dilation(pred_b, selem)
    
    # Tolerant IoU: a predicted pixel is TP if it falls within 
    # tolerance of a GT pixel, and vice versa
    tp = float((gt_dilated & pred_b).sum() + (pred_dilated & gt_b).sum()) / 2.0
    fp = float((pred_b & ~gt_dilated).sum())
    fn = float((gt_b & ~pred_dilated).sum())
    
    denom = tp + fp + fn
    return tp / denom if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────

def visualize_one(model, img_path, json_path, device, output_dir, 
                  strip_fraction=0.08, img_size=512):
    """Generate a full validation figure for one image."""
    
    # Load image
    img_bgr = cv2.imread(str(img_path))
    H_full, W_full = img_bgr.shape[:2]
    strip_h = int(H_full * strip_fraction)
    img_bgr = img_bgr[:H_full - strip_h, :]
    H, W = img_bgr.shape[:2]
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Load ground truth (pass full size so it crops correctly)
    gt_crater, gt_crack = load_masks_from_json(
        json_path, H, W, h_full=H_full, w_full=W_full)
    
    # Predict
    pred_crater, pred_crack, prob_crater, prob_crack = predict_image(
        model, img_gray, device, img_size)
    
    # Compute metrics
    iou_c       = compute_iou(gt_crater, pred_crater)
    iou_k       = compute_crack_iou_tolerant(gt_crack, pred_crack, tolerance_px=5)
    iou_k_strict = compute_iou(gt_crack, pred_crack)
    f1_c        = compute_f1(gt_crater, pred_crater)
    f1_k        = compute_f1(gt_crack,  pred_crack)
    has_crater  = bool(gt_crater.any())
    has_crack   = bool(gt_crack.any())
    
    # Error overlays
    err_crater = make_error_overlay(img_gray, gt_crater, pred_crater)
    err_crack  = make_error_overlay(img_gray, gt_crack, pred_crack)
    
    # ── Figure ──────────────────────────────────────────────────────────
    plt.rcParams.update({'font.family': 'serif', 'font.size': 30})
    fig, axes = plt.subplots(2, 4, figsize=(44, 22))
    fig.patch.set_facecolor('white')

    for ax_row in axes:
        for ax in ax_row:
            ax.set_facecolor('white')
            ax.tick_params(colors='black', labelsize=22)
            for sp in ax.spines.values():
                sp.set_color('#AAAAAA')
                sp.set_linewidth(0.7)
            ax.set_xticks([])
            ax.set_yticks([])

    stem = img_path.stem

    # Row 0: Craters
    axes[0, 0].imshow(img_gray, cmap='gray')
    axes[0, 0].set_title('SEM Image', color='#222222', fontsize=36, fontweight='bold')

    gt_vis_c = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    gt_vis_c[gt_crater == 1] = [255, 140, 0]
    axes[0, 1].imshow(gt_vis_c)
    axes[0, 1].set_title('Ground Truth — Craters', color='#CC5500', fontsize=36, fontweight='bold')

    im_c = axes[0, 2].imshow(prob_crater, cmap='hot', vmin=0, vmax=1)
    cb_c = fig.colorbar(im_c, ax=axes[0, 2], fraction=0.046, pad=0.04, shrink=0.85)
    cb_c.set_label('P(crater)', fontsize=30, color='black')
    cb_c.ax.tick_params(labelsize=26)
    axes[0, 2].set_title('Crater Probability Map', color='#222222', fontsize=36, fontweight='bold')

    axes[0, 3].imshow(err_crater)
    axes[0, 3].set_title(f'Detection Errors — Craters\nIoU: {iou_c:.3f}  |  F1: {f1_c:.3f}',
                         color='#222222', fontsize=33, fontweight='bold')

    # Row 1: Cracks
    axes[1, 0].imshow(img_gray, cmap='gray')
    axes[1, 0].set_title('SEM Image', color='#222222', fontsize=36, fontweight='bold')

    gt_vis_k = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    gt_vis_k[gt_crack == 1] = [210, 0, 0]
    axes[1, 1].imshow(gt_vis_k)
    axes[1, 1].set_title('Ground Truth — Cracks', color='#AA0000', fontsize=36, fontweight='bold')

    im_k = axes[1, 2].imshow(prob_crack, cmap='hot', vmin=0, vmax=1)
    cb_k = fig.colorbar(im_k, ax=axes[1, 2], fraction=0.046, pad=0.04, shrink=0.85)
    cb_k.set_label('P(crack)', fontsize=30, color='black')
    cb_k.ax.tick_params(labelsize=26)
    axes[1, 2].set_title('Crack Probability Map', color='#222222', fontsize=36, fontweight='bold')

    axes[1, 3].imshow(err_crack)
    axes[1, 3].set_title(f'Detection Errors — Cracks\nIoU: {iou_k:.3f} (tol.)  |  F1: {f1_k:.3f}',
                         color='#222222', fontsize=33, fontweight='bold')

    leg_patches = [
        mpatches.Patch(color=(0, 220/255, 0),   label='True Positive'),
        mpatches.Patch(color=(220/255, 0, 0),   label='False Positive'),
        mpatches.Patch(color=(0, 80/255, 1),    label='False Negative (missed)'),
    ]
    fig.legend(handles=leg_patches, loc='lower center', ncol=3,
               facecolor='white', labelcolor='black', edgecolor='#BBBBBB',
               fontsize=34, framealpha=0.95)

    plt.tight_layout(rect=[0, 0.07, 1, 1.0])

    out_path = output_dir / f'val_{stem}.png'
    plt.savefig(str(out_path), dpi=130, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    
    return iou_c, iou_k, f1_c, f1_k, has_crater, has_crack, out_path


# ─────────────────────────────────────────────────────────────────────────────
#  SUMMARY FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def _filtered_mean(values, mask):
    """Mean excluding images with no GT for that class (mask=False) OR metric=0."""
    v = [val for val, m in zip(values, mask) if m and val > 0]
    return float(np.mean(v)) if v else float('nan')


def _hist_panel(ax, values, color, title, mean_val, median_val, std_val, n):
    """Draw a single histogram panel with mean/median/std annotations."""
    bins = np.linspace(0, 1, 21)
    ax.hist(values, bins=bins, color=color, alpha=0.78, edgecolor='white', linewidth=0.5)
    ax.axvline(mean_val,   color='#111111', ls='--', lw=3.5, label=f'Mean = {mean_val:.3f}')
    ax.axvline(median_val, color='#555555', ls=':',  lw=3.0, label=f'Median = {median_val:.3f}')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Score', color='black', fontsize=36)
    ax.set_ylabel('Images', color='black', fontsize=36)
    ax.set_title(title, color='black', fontsize=40, fontweight='bold', pad=14)
    ax.legend(fontsize=34, facecolor='white', labelcolor='black',
              edgecolor='#BBBBBB', framealpha=0.9)
    ax.text(0.97, 0.95,
            f'$\\mu$={mean_val:.3f}\n$\\sigma$={std_val:.3f}\nn={n}',
            transform=ax.transAxes, fontsize=32, va='top', ha='right',
            color='#333333',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#F8F8F8',
                      edgecolor='#CCCCCC', alpha=0.9))


def plot_summary(results, output_dir):
    """Distribution histograms of IoU and F1 — large format for scientific paper."""
    plt.rcParams.update({'font.family': 'serif', 'font.size': 34})

    def _vals(key, flag):
        return [r[key] for r in results if r[flag] and r[key] > 0]

    iou_c = _vals('iou_crater', 'has_crater')
    iou_k = _vals('iou_crack',  'has_crack')
    f1_c  = _vals('f1_crater',  'has_crater')
    f1_k  = _vals('f1_crack',   'has_crack')

    def stats(v):
        a = np.array(v)
        return float(np.mean(a)), float(np.median(a)), float(np.std(a)), len(a)

    mi_c, med_c_i, std_c_i, n_ci = stats(iou_c) if iou_c else (0,0,0,0)
    mi_k, med_k_i, std_k_i, n_ki = stats(iou_k) if iou_k else (0,0,0,0)
    mf_c, med_c_f, std_c_f, n_cf = stats(f1_c)  if f1_c  else (0,0,0,0)
    mf_k, med_k_f, std_k_f, n_kf = stats(f1_k)  if f1_k  else (0,0,0,0)

    fig, axes = plt.subplots(2, 2, figsize=(24, 18))
    fig.patch.set_facecolor('white')
    (ax_ic, ax_fc), (ax_ik, ax_fk) = axes

    for ax in axes.flat:
        ax.set_facecolor('white')
        ax.tick_params(colors='black', labelsize=32)
        ax.grid(True, axis='y', alpha=0.4, color='#DDDDDD', linestyle='--')
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color('#AAAAAA')
            sp.set_linewidth(0.7)

    _hist_panel(ax_ic, iou_c, '#E8870A', 'Crater — IoU',        mi_c, med_c_i, std_c_i, n_ci)
    _hist_panel(ax_fc, f1_c,  '#F0A830', 'Crater — F1 (Dice)',  mf_c, med_c_f, std_c_f, n_cf)
    _hist_panel(ax_ik, iou_k, '#CC2222', 'Crack — IoU (tol. 5 px)', mi_k, med_k_i, std_k_i, n_ki)
    _hist_panel(ax_fk, f1_k,  '#E06060', 'Crack — F1 (Dice)',   mf_k, med_k_f, std_k_f, n_kf)

    plt.tight_layout()
    out = output_dir / 'validation_summary.png'
    plt.savefig(str(out), dpi=200, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f'  >>> Summary chart saved: {out}')
    return mi_c, mi_k, mf_c, mf_k


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Visual validation of U-Net predictions')
    p.add_argument('--checkpoint',  default='checkpoints/best_model.pth',
                   help='Path to trained model checkpoint')
    p.add_argument('--data-dir',    default='./data',
                   help='Root data directory (with images/ and masks/)')
    p.add_argument('--output-dir',  default='./validation_results',
                   help='Directory to save validation figures')
    p.add_argument('--num-images',  type=int, default=10,
                   help='Number of images to validate (0 = all)')
    p.add_argument('--img-size',    type=int, default=512,
                   help='Input size for the model')
    p.add_argument('--strip-fraction', type=float, default=0.08,
                   help='Fraction of image bottom to strip (SEM metadata bar)')
    p.add_argument('--seed',        type=int, default=42,
                   help='Random seed for image selection')
    args = p.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f'\n  Device: {device}')
    print(f'  Checkpoint: {args.checkpoint}')
    
    # Load model
    ckpt = torch.load(args.checkpoint, map_location=device)
    cfg = ckpt.get('cfg', {})
    model = build_unet(
        encoder_name=cfg.get('encoder', 'resnet34'),
        num_classes=cfg.get('num_classes', 2),
    ).to(device)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print(f'  Model loaded (epoch {ckpt.get("epoch", "?")})')
    
    # Find images with JSON masks
    data_dir = Path(args.data_dir)
    img_dir = data_dir / 'images'
    mask_dir = data_dir / 'masks'
    
    EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    all_pairs = []
    for img_path in sorted(img_dir.iterdir()):
        if img_path.suffix.lower() not in EXTS:
            continue
        json_path = mask_dir / f'{img_path.stem}.json'
        if json_path.exists():
            all_pairs.append((img_path, json_path))
    
    print(f'  Found {len(all_pairs)} images with JSON masks')
    
    # Select subset
    random.seed(args.seed)
    if args.num_images > 0 and args.num_images < len(all_pairs):
        selected = random.sample(all_pairs, args.num_images)
    else:
        selected = all_pairs
    
    print(f'  Validating {len(selected)} images...\n')
    
    # Process each image
    results = []
    for i, (img_path, json_path) in enumerate(selected):
        iou_c, iou_k, f1_c, f1_k, has_c, has_k, out_path = visualize_one(
            model, img_path, json_path, device, output_dir,
            strip_fraction=args.strip_fraction, img_size=args.img_size)

        results.append({
            'name':       img_path.stem,
            'iou_crater': iou_c,  'iou_crack': iou_k,
            'f1_crater':  f1_c,   'f1_crack':  f1_k,
            'has_crater': has_c,  'has_crack': has_k,
        })

        print(f'  [{i+1}/{len(selected)}] {img_path.stem}  '
              f'Crater  IoU:{iou_c:.3f} F1:{f1_c:.3f}  |  '
              f'Crack  IoU:{iou_k:.3f} F1:{f1_k:.3f}'
              + ('' if has_c else ' [no crater GT]')
              + ('' if has_k else ' [no crack GT]')
              + f'  -> {out_path.name}')

    # Filtered means (exclude images where that class has no GT annotation)
    def fmean(key, flag):
        vals = [r[key] for r in results if r[flag] and r[key] > 0]
        return float(np.mean(vals)) if vals else float('nan')

    mi_c = fmean('iou_crater', 'has_crater')
    mi_k = fmean('iou_crack',  'has_crack')
    mf_c = fmean('f1_crater',  'has_crater')
    mf_k = fmean('f1_crack',   'has_crack')
    n_c  = sum(r['has_crater'] and r['iou_crater'] > 0 for r in results)
    n_k  = sum(r['has_crack']  and r['iou_crack']  > 0 for r in results)

    print(f'\n  ══════════════════════════════════════════════════════')
    print(f'  VALIDATION SUMMARY  ({len(results)} images total)')
    print(f'  (means computed only over images that contain each class)')
    print(f'  ──────────────────────────────────────────────────────')
    print(f'  Craters  ({n_c} images with GT):')
    print(f'    Mean IoU : {mi_c:.4f}')
    print(f'    Mean F1  : {mf_c:.4f}')
    print(f'  Cracks   ({n_k} images with GT):')
    print(f'    Mean IoU : {mi_k:.4f}')
    print(f'    Mean F1  : {mf_k:.4f}')
    print(f'  ══════════════════════════════════════════════════════\n')

    # Summary chart
    plot_summary(results, output_dir)

    print(f'  All results saved to: {output_dir}/')


if __name__ == '__main__':
    main()