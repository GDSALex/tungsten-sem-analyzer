"""
=============================================================================
inference.py  —  Complete inference pipeline
=============================================================================
Combines U-Net segmentation with classical post-processing to produce
publication-ready analysis of tungsten SEM images.

PIPELINE (revision 1 of the manuscript):
  1. Load image; region of interest = frame without its bottom 8 % of rows (the crop
     used in training), further restricted to the rows above the SEM data bar
  2. Scale: pixel size read from the data bar and accepted only when the printed
     magnification and field width agree within 1 % (or given with --scale); otherwise
     every physical output is withheld and the image is flagged
  3. U-Net prediction (craters + cracks), unchanged deployed pipeline
  4. Analysis stage: crater like regions (areas, equivalent diameters, neutral size
     category of the largest one) and cracks (revised centreline, Kulpa length,
     edge connected / near edge / internal, spacing CV)
  5. Figures: original vs segmentation (edge row drawn), comparison with a manual
     annotation when one is available
  The size categories are size classes of the largest crater like region only.  They
  are not calibrated against depth, material loss or any other physical endpoint.

USAGE:
  python -m tungsten_analyzer.inference                                # Interactive mode
  python -m tungsten_analyzer.inference --input image.jpg              # Single image
  python -m tungsten_analyzer.inference --input ./folder --batch       # Batch mode
  python -m tungsten_analyzer.inference --input image.jpg --gt mask.json   # With annotation
  python -m tungsten_analyzer.inference --input image.jpg --scale 0.2236   # Pixel size given

OUTPUT (per image):
  - {name}_segmentation.png    : Original vs U-Net segmentation
  - {name}_gt_comparison.png   : annotation vs prediction (if a JSON is provided)
  - {name}_report.json         : Full quantitative report, with the pixel size and its source
  - summary.csv                : Batch summary (batch mode only)
=============================================================================
"""

import argparse
import json
import csv
import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, List

import numpy as np
import cv2
import torch
import torch.nn.functional as F
from skimage.morphology import skeletonize, binary_dilation, disk, remove_small_objects
from skimage.measure import regionprops, label as ski_label

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from tungsten_analyzer.model import build_unet, build_unet_deep_supervised
from tungsten_analyzer import measurement as MS

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

# Neutral size categories of the largest crater like region (equivalent diameter).
# The first version called them severity grades, with labels from "no structural damage"
# to "catastrophic damage"; they were never calibrated against any physical endpoint.
SIZE_CATEGORIES = MS.SIZE_CATEGORIES

STRIP_FRACTION = 0.08      # bottom rows removed in training and inference

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# JSON mask labels
_CRATER_LABELS = {'crater', 'cratera', 'cráter', 'vaporization',
                  'collapse', 'crater_vaporization', 'crater_collapse'}
_CRACK_LABELS  = {'crack', 'grieta', 'fracture', 'fractura',
                  'crack_large', 'crack_small'}


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSICAL INFRASTRUCTURE (mirrors classical_pipeline.py)
# ─────────────────────────────────────────────────────────────────────────────

def detect_meta_strip(gray):
    """Height of the data bar found by the detector of the first version.

    Kept for reference only: it removes 76 rows on the OLMAT images, whereas training
    removed 61 (8 %).  The revised analysis uses the training crop and
    `measurement.locate_data_bar` instead.
    """
    h = gray.shape[0]
    for frac in (0.15, 0.12, 0.10, 0.08):
        s = int(h * frac)
        strip = gray[h - s:, :]
        if strip.mean() < 110:
            _, bw = cv2.threshold(strip, 190, 255, cv2.THRESH_BINARY)
            if bw.sum() / 255 > 30:
                return s
    return int(h * 0.10)


def _configure_tesseract():
    """Use the default Windows install path when tesseract is not on the PATH."""
    try:
        import shutil
        import pytesseract
        default = Path(r'C:\Program Files\Tesseract-OCR\tesseract.exe')
        if shutil.which('tesseract') is None and default.exists():
            pytesseract.pytesseract.tesseract_cmd = str(default)
    except ImportError:
        pass


def read_scale(gray_full):
    """Verified pixel size of a full frame (dict from `measurement.read_scale`).

    The first version read only the integer part of the magnification (``1.50 kX`` was
    read as 1x), converted it with a constant of 111.8 um and, when nothing was read,
    fell back silently on 0.224 um/px.  The pixel size is now accepted only when the
    printed magnification and field width agree within 1 %.
    """
    _configure_tesseract()
    return MS.read_scale(gray_full)


def auto_scale(gray, meta_h=None):
    """Verified pixel size in um/px, or None (never a default value).

    Kept with its old signature for the scripts that call it; `meta_h` is ignored.
    """
    res = read_scale(gray)
    if res['um_per_px'] is None:
        print(f"  [Scale] not verified: {res['reason']}")
    return res['um_per_px']


def compute_zones_dynamic(roi_gray, bottom_pct=0.02):
    from scipy import ndimage
    h, w = roi_gray.shape
    artifact_start = int(h * (1.0 - bottom_pct))
    max_search = int(h * 0.35)
    search_roi = roi_gray[:max_search, :]
    blur = cv2.GaussianBlur(search_roi, (9, 9), 0)
    sobel_y = np.abs(cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5))
    sobel_x = np.abs(cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5))
    texture_map = sobel_x + sobel_y
    texture_profile = np.mean(texture_map, axis=1)
    texture_smooth = ndimage.gaussian_filter1d(texture_profile, sigma=3.0)
    safe_top = 10
    if len(texture_smooth) > safe_top:
        grad_texture = np.diff(texture_smooth)
        max_grad = np.max(grad_texture[safe_top:])
        max_t, min_t = np.max(texture_smooth), np.min(texture_smooth)
        umbral_material = min_t + (max_t - min_t) * 0.40
        picos = np.where(grad_texture[safe_top:] > max_grad * 0.35)[0] + safe_top
        peak_edge = None
        for pico in reversed(picos):
            if pico + 10 < len(texture_smooth) and texture_smooth[pico + 10] > umbral_material:
                peak_edge = pico
                break
        if peak_edge is None:
            peak_edge = int(np.argmax(grad_texture[safe_top:])) + safe_top
        if (max_t - min_t) > 10.0 and texture_smooth[peak_edge] > min_t + 5.0:
            edge_end = peak_edge + 3
        else:
            edge_end = 0
    else:
        edge_end = 0
    return int(min(max_search, edge_end)), int(artifact_start)


# ─────────────────────────────────────────────────────────────────────────────
#  U-NET INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

class UNetPredictor:
    def __init__(self, checkpoint_path, device=None, img_size=512):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        self.img_size = img_size
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        cfg = ckpt.get('cfg', {})

        # Detect if checkpoint is from a deep supervised model
        is_ds = cfg.get('use_deep_supervision', False)
        # Also check state_dict keys for 'base.' prefix (DeepSupervisedUNet)
        state_keys = list(ckpt['model_state'].keys())
        if any(k.startswith('base.') for k in state_keys):
            is_ds = True

        if is_ds:
            self.model = build_unet_deep_supervised(
                encoder_name=cfg.get('encoder', 'resnet34'),
                encoder_weights=None,
                num_classes=cfg.get('num_classes', 2),
            ).to(self.device)
            print(f"  [U-Net] Deep Supervised model detected")
        else:
            self.model = build_unet(
                encoder_name=cfg.get('encoder', 'resnet34'),
                encoder_weights=None,
                num_classes=cfg.get('num_classes', 2),
            ).to(self.device)

        self.model.load_state_dict(ckpt['model_state'])
        self.model.eval()
        epoch = ckpt.get('epoch', '?')
        print(f"  [U-Net] Model loaded (epoch {epoch}) | Device: {self.device}")

    @torch.no_grad()
    def predict(self, gray, threshold_crater=0.5, threshold_crack=0.25,
                hysteresis_low=0.08, hysteresis_high=0.20):
        """
        Multi-scale inference combining:
          1. Global pass (full image → 512): good for craters, global context
          2. Sliding-window pass (tiles of 512 at higher resolution): good for fine cracks
          3. Edge-zone pass at higher overlap for crack-dense regions
          4. TTA (4 flips) on each pass
          5. Hysteresis thresholding for cracks: recovers faint crack pixels
             connected to confident ones
          6. Morphological crack bridging: reconnects broken segments

        The crack channel uses the maximum probability across scales (any scale
        that sees the crack is enough). Craters use the global pass primarily.
        """
        H, W = gray.shape[:2]
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB) if gray.ndim == 2 else gray

        # ── PASS 1: Global (same as before) ──────────────────────────
        global_probs = self._predict_single(rgb, H, W)

        # ── PASS 2: Sliding window at higher effective resolution ─────
        tile_probs = self._predict_tiled(rgb, H, W)

        # ── MERGE: max across scales for cracks, weighted for craters ─
        crater_prob = global_probs[0]

        if tile_probs is not None:
            crack_prob = np.maximum(global_probs[1], tile_probs[1])
            crater_prob = np.maximum(crater_prob, tile_probs[0] * 0.7)
        else:
            crack_prob = global_probs[1]

        # ── PASS 3: Edge-zone high-overlap pass ──────────────────────
        # Cracks concentrate near the material edge (top portion).
        # Re-tile that region with higher overlap for finer detail.
        edge_fraction = 0.30
        edge_h = int(H * edge_fraction)
        if edge_h >= self.img_size:
            edge_probs = self._predict_tiled(
                rgb[:edge_h, :], edge_h, W, overlap=0.6)
            if edge_probs is not None:
                crack_prob[:edge_h, :] = np.maximum(
                    crack_prob[:edge_h, :], edge_probs[1])
                crater_prob[:edge_h, :] = np.maximum(
                    crater_prob[:edge_h, :], edge_probs[0] * 0.5)

        # ── RIDGE ENHANCEMENT on crack probability map ───────────────
        # Apply oriented filter to boost thin linear structures in the
        # probability map before thresholding. This helps recover faint
        # cracks that are below threshold but have clear ridge structure.
        crack_prob = self._enhance_ridges(crack_prob)

        # ── HYSTERESIS THRESHOLDING for cracks ───────────────────────
        # Standard binary threshold loses faint crack pixels. Hysteresis
        # uses two thresholds: "high" seeds confident crack pixels, "low"
        # extends from those seeds into neighboring faint pixels.
        # This is the standard approach in edge detection (Canny) and
        # crack/vessel segmentation.
        crack_mask = _hysteresis_threshold(
            crack_prob, low=hysteresis_low, high=hysteresis_high)

        # ── MORPHOLOGICAL CRACK BRIDGING ─────────────────────────────
        # Fine cracks often break into fragments due to noise or resolution
        # limits. Bridge small gaps (≤15px) along the crack direction.
        crack_mask = _bridge_crack_gaps(crack_mask, max_gap=15)

        crater_mask = (crater_prob > threshold_crater).astype(bool)

        return crater_mask, crack_mask, crater_prob, crack_prob

    @torch.no_grad()
    def _predict_single(self, rgb, H, W):
        """Single-scale prediction with TTA (4 flips)."""
        resized = cv2.resize(rgb, (self.img_size, self.img_size)).astype(np.float32) / 255.0
        normed = (resized - _MEAN) / _STD
        tensor = torch.from_numpy(normed.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)

        probs_accum = torch.zeros(1, 2, self.img_size, self.img_size, device=self.device)
        for fh in [False, True]:
            for fv in [False, True]:
                inp = tensor.clone()
                if fh: inp = torch.flip(inp, [3])
                if fv: inp = torch.flip(inp, [2])
                p = torch.sigmoid(self.model(inp))
                if fv: p = torch.flip(p, [2])
                if fh: p = torch.flip(p, [3])
                probs_accum += p
        probs = (probs_accum / 4.0)[0].cpu().numpy()

        prob_c = cv2.resize(probs[0], (W, H), interpolation=cv2.INTER_LINEAR)
        prob_k = cv2.resize(probs[1], (W, H), interpolation=cv2.INTER_LINEAR)
        return np.stack([prob_c, prob_k])

    @torch.no_grad()
    def _predict_tiled(self, rgb, H, W, overlap=0.5, min_scale_ratio=1.3):
        """
        Sliding-window inference at higher effective resolution.
        
        Splits the image into overlapping tiles of img_size, processes each,
        and stitches using Gaussian-weighted blending.
        
        Only runs if the image is at least min_scale_ratio× larger than
        img_size in either dimension (otherwise global pass is sufficient).
        """
        sz = self.img_size
        max_dim = max(H, W)
        
        # Skip tiling if image is already close to model resolution
        if max_dim < sz * min_scale_ratio:
            return None
        
        # Determine tile scale: we want tiles that cover ~512px of the
        # original image, so each tile sees 2× the detail vs global pass.
        # For a 1500px wide image at sz=512, tile_step ≈ 384px with 25% overlap.
        stride = int(sz * (1 - overlap))
        
        # Accumulation buffers
        prob_sum   = np.zeros((2, H, W), dtype=np.float64)
        weight_sum = np.zeros((H, W), dtype=np.float64)
        
        # Gaussian weighting kernel (reduces edge artifacts in stitching)
        gauss_1d = np.exp(-0.5 * ((np.arange(sz) - sz/2) / (sz/4))**2)
        gauss_2d = np.outer(gauss_1d, gauss_1d).astype(np.float64)
        
        # Generate tile positions
        y_positions = list(range(0, max(1, H - sz + 1), stride))
        if y_positions[-1] + sz < H:
            y_positions.append(H - sz)
        x_positions = list(range(0, max(1, W - sz + 1), stride))
        if x_positions[-1] + sz < W:
            x_positions.append(W - sz)
        
        for y in y_positions:
            for x in x_positions:
                # Extract tile
                y2, x2 = min(y + sz, H), min(x + sz, W)
                y1, x1 = y2 - sz, x2 - sz  # ensure exact sz×sz
                
                tile = rgb[max(0,y1):y2, max(0,x1):x2]
                th, tw = tile.shape[:2]
                
                if th < sz or tw < sz:
                    tile = cv2.resize(tile, (sz, sz), interpolation=cv2.INTER_LINEAR)
                
                # Normalize
                tile_f = tile.astype(np.float32) / 255.0
                tile_n = (tile_f - _MEAN) / _STD
                tensor = torch.from_numpy(tile_n.transpose(2, 0, 1)).unsqueeze(0).float().to(self.device)
                
                # Predict (simplified TTA: just horizontal flip for speed)
                probs_accum = torch.zeros(1, 2, sz, sz, device=self.device)
                for fh in [False, True]:
                    for fv in [False, True]:
                        inp = tensor.clone()
                        if fh: inp = torch.flip(inp, [3])
                        if fv: inp = torch.flip(inp, [2])
                        p = torch.sigmoid(self.model(inp))
                        if fv: p = torch.flip(p, [2])
                        if fh: p = torch.flip(p, [3])
                        probs_accum += p
                probs = (probs_accum / 4.0)[0].cpu().numpy()
                
                # Resize back if tile was resized
                if th < sz or tw < sz:
                    probs_resized = np.stack([
                        cv2.resize(probs[c], (tw, th), interpolation=cv2.INTER_LINEAR)
                        for c in range(2)])
                else:
                    probs_resized = probs
                
                # Gaussian weight for this tile region
                gw = gauss_2d[:th, :tw] if (th == sz and tw == sz) else \
                     cv2.resize(gauss_2d, (tw, th), interpolation=cv2.INTER_LINEAR)
                
                # Accumulate
                py1, py2 = max(0, y1), max(0, y1) + th
                px1, px2 = max(0, x1), max(0, x1) + tw
                prob_sum[:, py1:py2, px1:px2] += probs_resized * gw[np.newaxis]
                weight_sum[py1:py2, px1:px2]  += gw
        
        # Normalize by weights
        weight_sum = np.maximum(weight_sum, 1e-8)
        result = prob_sum / weight_sum[np.newaxis]
        
        return result

    def _enhance_ridges(self, prob_map, sigma=1.5, beta=0.5):
        """
        Enhance thin linear structures (cracks) in the probability map
        using eigenvalues of the Hessian matrix (Frangi-like filter).

        This boosts faint cracks that have clear ridge structure but low
        overall probability, without boosting blob-like noise.
        """
        from scipy.ndimage import gaussian_filter

        # Compute Hessian components
        smooth = gaussian_filter(prob_map.astype(np.float64), sigma=sigma)
        Hxx = gaussian_filter(smooth, sigma=sigma, order=[0, 2])
        Hyy = gaussian_filter(smooth, sigma=sigma, order=[2, 0])
        Hxy = gaussian_filter(smooth, sigma=sigma, order=[1, 1])

        # Eigenvalue-based ridge measure: for a ridge (crack), one eigenvalue
        # is large negative (across the ridge) and the other is ~0 (along it).
        # We want pixels where the smaller eigenvalue magnitude is much less
        # than the larger one (tubular structure, not blob).
        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy ** 2
        discriminant = np.maximum(trace ** 2 - 4 * det, 0)
        sqrt_disc = np.sqrt(discriminant)

        lambda1 = (trace + sqrt_disc) / 2.0
        lambda2 = (trace - sqrt_disc) / 2.0

        # Ridge response: strong negative lambda (concave = dark crack in
        # bright background, but in probability space: bright ridge on dark bg)
        # We take the most negative eigenvalue; for bright ridges on dark bg,
        # the second eigenvalue is large positive.
        ridge = np.maximum(-lambda2, 0)

        # Normalize to [0, 1]
        ridge_max = ridge.max()
        if ridge_max > 1e-8:
            ridge = ridge / ridge_max

        # Blend: boost probability where there's ridge structure
        enhanced = prob_map + beta * ridge * (1.0 - prob_map)
        return np.clip(enhanced, 0, 1).astype(np.float64)


# ─────────────────────────────────────────────────────────────────────────────
#  CRACK-SPECIFIC POST-PROCESSING
# ─────────────────────────────────────────────────────────────────────────────

def _hysteresis_threshold(prob_map, low=0.08, high=0.20):
    """
    Two-threshold (hysteresis) binarization for crack probability maps.

    Standard approach from Canny edge detection, adapted for segmentation:
      1. Pixels above `high` are definite cracks (seeds).
      2. Pixels above `low` but below `high` are tentative.
      3. Tentative pixels are accepted ONLY if connected to a seed pixel.

    This recovers faint sections of cracks that are connected to confident
    detections, without accepting isolated noise above the low threshold.
    """
    from scipy.ndimage import label as ndi_label

    strong = prob_map >= high
    weak = (prob_map >= low) & ~strong

    # Label connected components in the union of strong + weak
    combined = strong | weak
    labeled, n_features = ndi_label(combined)

    # Keep only components that contain at least one strong pixel
    if n_features == 0:
        return strong

    # Find which labels contain strong pixels
    strong_labels = set(np.unique(labeled[strong]))
    strong_labels.discard(0)

    # Build output: accept all pixels in components that have a strong seed
    result = np.zeros_like(prob_map, dtype=bool)
    for lab in strong_labels:
        result |= (labeled == lab)

    return result


def _bridge_crack_gaps(crack_mask, max_gap=15):
    """
    Bridge small gaps in crack masks using morphological closing with
    oriented structuring elements.

    Fine cracks often break into disconnected fragments due to:
      - Sub-pixel width at the model's resolution
      - Noise/texture that locally suppresses the probability
      - Image artifacts

    This applies directional closing at multiple angles (0°, 45°, 90°, 135°)
    to reconnect fragments that are nearly collinear within `max_gap` pixels.

    Note (revision 1): every pixel added by the closings is accepted.  No check is made
    that a new pixel joins two existing crack regions, as the first version of this
    docstring stated, so the closing can also fill concavities and thicken single
    fragments.  On the out-of-fold predictions the bridged pixels add about 5 % of the
    centreline length, and only about 3 % of that added length lies within 5 px of an
    annotated crack (Section S5.4 of the supplementary material of the revised
    manuscript).  The behaviour is kept unchanged so that the deployed pipeline is the
    one evaluated in the paper.
    """
    if not crack_mask.any():
        return crack_mask

    result = crack_mask.copy()

    # Create oriented line structuring elements at 4 angles
    for angle in [0, 45, 90, 135]:
        se = _oriented_line_se(max_gap, angle)
        # Morphological closing: dilate then erode
        closed = cv2.morphologyEx(
            result.astype(np.uint8), cv2.MORPH_CLOSE, se)
        # Only accept new pixels that bridge between existing crack regions:
        # a bridged pixel must be new (not in original mask) AND must connect
        # to the original mask on both sides along the SE direction.
        new_pixels = (closed > 0) & ~result
        if new_pixels.any():
            # All new pixels are accepted (no bridging check, see the docstring)
            result = result | new_pixels

    # Remove any small isolated dots introduced by bridging
    result = remove_small_objects(result, min_size=5)

    return result


def _oriented_line_se(length, angle_deg):
    """Create a line structuring element at a given angle."""
    rad = np.deg2rad(angle_deg)
    half = length // 2
    size = 2 * half + 1
    se = np.zeros((size, size), dtype=np.uint8)

    for i in range(-half, half + 1):
        y = int(round(half + i * np.sin(rad)))
        x = int(round(half + i * np.cos(rad)))
        y = np.clip(y, 0, size - 1)
        x = np.clip(x, 0, size - 1)
        se[y, x] = 1

    return se


# ─────────────────────────────────────────────────────────────────────────────
#  POST-PROCESSING & METRICS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_craters(crater_mask, roi_gray, um_per_px, edge_end, artifact_start):
    """Crater like regions: label, filter, measure.

    Returns (details, size category of the largest region, label image).  The region
    filters are those of the first version.  With `um_per_px=None` (scale not verified)
    only pixel measurements are returned and the size category is None.
    """
    # Remove small noise
    clean = remove_small_objects(crater_mask, min_size=300)
    
    # Exclude edge zone and artifact zone
    if edge_end > 0:
        clean[:edge_end, :] = False
    if artifact_start > 0 and artifact_start < clean.shape[0]:
        clean[artifact_start:, :] = False

    labels = ski_label(clean)
    details = []
    
    for pp in regionprops(labels, intensity_image=roi_gray):
        area_px = int(pp.area)
        if area_px < 300:
            continue
        
        # Filter artifacts (bottom zone)
        min_r = pp.bbox[0]
        if min_r > artifact_start * 0.45:
            if area_px < 600:
                continue

        diam_px = float(getattr(pp, 'equivalent_diameter_area',
                                getattr(pp, 'equivalent_diameter', 0.0)))
        scaled = um_per_px is not None
        details.append({
            'label': pp.label,
            'area_px': area_px,
            'diam_px': round(diam_px, 1),
            'area_um2': round(area_px * um_per_px ** 2, 1) if scaled else None,
            'diam_um': round(diam_px * um_per_px, 1) if scaled else None,
            'perimeter_um': round(pp.perimeter * um_per_px, 1) if scaled else None,
            'centroid': (float(pp.centroid[0]), float(pp.centroid[1])),
            'bbox': tuple(int(v) for v in pp.bbox),
            'eccentricity': round(float(pp.eccentricity), 3),
            'solidity': round(float(pp.solidity), 3),
        })

    if um_per_px is None:
        return details, None, labels
    largest = max((d['diam_um'] for d in details), default=None)
    return details, MS.size_category(largest), labels


def analyze_cracks(crack_mask, um_per_px, edge_end):
    """Revised crack analysis (see `measurement.analyze_cracks`).

    Returns (crack_info, centreline skeleton, label image of the kept cracks).  The
    length of each crack is the Kulpa length of its pruned centreline, which replaces
    the skeleton pixel count of the first version (`analyze_cracks_submitted`).
    """
    from scipy import ndimage as ndi
    info = MS.analyze_cracks(crack_mask.astype(bool), um_per_px, edge_end)
    skel = info.pop('skeleton')
    labels, _ = ndi.label(skel, structure=np.ones((3, 3), int))
    return info, skel, labels


_CRACK_CLASSES_SUBMITTED = {
    'large': {0: 'None', 1: '< 1000 µm', 2: '≥ 1000 µm'},
    'small': {0: 'None', 1: '1–2 cracks', 2: '> 2 cracks'},
    'regular': {0: 'No', 1: 'Yes (CV < 45%)'},
}


def analyze_cracks_submitted(crack_mask, um_per_px, edge_end):
    """Crack analysis of the first version (skeleton pixel count), kept for reference.

    Its length estimate depends on the width of the mask: applied to the 15-px bands of
    the annotation it returns about twice the annotated length.
    """
    # Skeletonize
    skel = skeletonize(crack_mask.astype(bool))
    skel_labels = ski_label(skel)
    
    # Filter short skeletons
    min_skel_px = max(10, int(15.0 / um_per_px))
    skel_clean = np.zeros_like(skel, dtype=bool)
    labels_clean = np.zeros_like(skel_labels, dtype=np.int32)
    new_id = 1
    
    for pp in regionprops(skel_labels):
        if pp.area < min_skel_px:
            continue
        r, c = pp.coords[:, 0], pp.coords[:, 1]
        skel_clean[r, c] = True
        labels_clean[r, c] = new_id
        new_id += 1

    # Classify each crack
    large_list, small_list = [], []
    all_cracks = []
    
    for pp in regionprops(labels_clean):
        r, c = np.where(labels_clean == pp.label)
        skel_px = int(skel_clean[r, c].sum())
        length_um = skel_px * um_per_px
        connects_edge = bool(np.any(r < edge_end))
        
        entry = {
            'id': pp.label,
            'length_px': skel_px,
            'length_um': round(length_um, 1),
            'connects_edge': connects_edge,
            'centroid': (float(pp.centroid[0]), float(pp.centroid[1])),
            'bbox': tuple(int(v) for v in pp.bbox),
            'type': 'large (edge)' if connects_edge else 'small (internal)',
        }
        all_cracks.append(entry)
        (large_list if connects_edge else small_list).append(entry)

    # Crack classification
    large_max_um = max((x['length_um'] for x in large_list), default=0.0)
    large_cls = 0 if large_max_um < 1 else (1 if large_max_um < 1000 else 2)
    
    n_small = len(small_list)
    small_cls = 0 if n_small == 0 else (1 if n_small <= 2 else 2)

    # Regularity: surface X positions + relaxed CV threshold
    # Includes near-surface small cracks, not just "large" ones connected to edge
    regular_cls = 0
    regular_cv = None
    surface_margin = max(40, int(0.08 * crack_mask.shape[0]))

    near_surface = [
        x for x in all_cracks
        if x['connects_edge'] or x['bbox'][0] < (edge_end + surface_margin)
    ]

    if len(near_surface) >= 3:
        surface_x = []
        for crack in near_surface:
            lbl = crack['id']
            ys, xs = np.where(labels_clean == lbl)
            if len(ys) == 0:
                continue
            min_row = int(np.min(ys))
            top_band = ys <= (min_row + 10)
            surface_x.append(float(np.median(xs[top_band])))

        if len(surface_x) >= 3:
            x_sorted = sorted(surface_x)
            spacings = np.diff(x_sorted)
            mean_sp = float(np.mean(spacings))
            if mean_sp > 1:
                cv_val = float(np.std(spacings) / (mean_sp + 1e-9))
                regular_cv = round(cv_val, 3)
                if cv_val < 0.45:
                    regular_cls = 1

    total_length_um = sum(x['length_um'] for x in all_cracks)
    max_crack_um = max((x['length_um'] for x in all_cracks), default=0.0)

    crack_info = {
        'large_crack_class': large_cls,
        'large_crack_class_desc': _CRACK_CLASSES_SUBMITTED['large'][large_cls],
        'large_crack_max_um': round(large_max_um, 1),
        'small_crack_class': small_cls,
        'small_crack_class_desc': _CRACK_CLASSES_SUBMITTED['small'][small_cls],
        'regular_crack_class': regular_cls,
        'regular_crack_class_desc': _CRACK_CLASSES_SUBMITTED['regular'][regular_cls],
        'regular_cv': regular_cv,
        'n_total': len(all_cracks),
        'n_large': len(large_list),
        'n_small': n_small,
        'total_length_um': round(total_length_um, 1),
        'max_single_crack_um': round(max_crack_um, 1),
        'individual_cracks': all_cracks,
    }
    
    return crack_info, skel_clean, labels_clean


# ─────────────────────────────────────────────────────────────────────────────
#  GROUND TRUTH LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_gt_masks(json_path, h, w, h_full, w_full):
    from PIL import Image, ImageDraw
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_w = data.get('imageWidth', w_full)
    img_h = data.get('imageHeight', h_full)
    canvas_c = Image.new('L', (img_w, img_h), 0)
    canvas_k = Image.new('L', (img_w, img_h), 0)
    draw_c, draw_k = ImageDraw.Draw(canvas_c), ImageDraw.Draw(canvas_k)
    crack_lw = max(7, int(15 * max(img_w, img_h) / 1024))
    for s in data.get('shapes', []):
        label = s['label'].strip().lower()
        pts = [(float(p[0]), float(p[1])) for p in s['points']]
        st = s.get('shape_type', 'polygon')
        draw = draw_c if label in _CRATER_LABELS else (draw_k if label in _CRACK_LABELS else None)
        if draw is None: continue
        if st == 'polygon' and len(pts) >= 3: draw.polygon(pts, fill=255)
        elif st in ('linestrip', 'line') and len(pts) >= 2: draw.line(pts, fill=255, width=crack_lw)
    mc, mk = np.array(canvas_c), np.array(canvas_k)
    if mc.shape[0] > h: mc, mk = mc[:h, :], mk[:h, :]
    if mc.shape[1] != w:
        mc = cv2.resize(mc, (w, mc.shape[0]), interpolation=cv2.INTER_NEAREST)
        mk = cv2.resize(mk, (w, mk.shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mc > 127).astype(np.uint8), (mk > 127).astype(np.uint8)


def load_gt_crack_thin(json_path, h, w, h_full, w_full):
    """Return a 1-px-wide crack mask drawn directly from linestrip points.

    The JSON linestrips are already skeletal paths, so drawing them at
    width=1 avoids the branching artifacts that appear when skeletonizing
    a thick painted mask. Used only for visualization, not IoU computation.
    """
    from PIL import Image, ImageDraw
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    img_w = data.get('imageWidth', w_full)
    img_h = data.get('imageHeight', h_full)
    canvas = Image.new('L', (img_w, img_h), 0)
    draw = ImageDraw.Draw(canvas)
    for s in data.get('shapes', []):
        label = s['label'].strip().lower()
        if label not in _CRACK_LABELS:
            continue
        pts = [(float(p[0]), float(p[1])) for p in s['points']]
        st = s.get('shape_type', 'polygon')
        if st in ('linestrip', 'line') and len(pts) >= 2:
            draw.line(pts, fill=255, width=1)
    mk = np.array(canvas)
    if mk.shape[0] > h:
        mk = mk[:h, :]
    if mk.shape[1] != w:
        mk = cv2.resize(mk, (w, mk.shape[0]), interpolation=cv2.INTER_NEAREST)
    return (mk > 0).astype(bool)


def compute_iou(gt, pred):
    gt_b, pred_b = gt.astype(bool), pred.astype(bool)
    tp = float((gt_b & pred_b).sum())
    denom = float((gt_b | pred_b).sum())
    return tp / denom if denom > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
#  FIGURE GENERATION
# ─────────────────────────────────────────────────────────────────────────────

BG = 'white'
C_CRATER_VIS = (0, 160, 230)   # Orange BGR
C_CRACK_VIS  = (0, 0, 210)     # Red BGR
C_SKEL_VIS   = (0, 200, 200)   # Teal BGR


C_CRACK_LABEL_VIS = {'edge connected': (0, 0, 210),      # red (BGR)
                     'near edge': (210, 0, 210),         # magenta
                     'internal': (0, 210, 230)}          # yellow
C_EDGE_ROW_VIS = (255, 200, 0)                           # cyan


def build_segmentation_overlay(img_gray, crater_mask, crack_labels, crack_info, edge_end,
                               crater_details=()):
    """Retained crater like regions (orange, numbered), cracks coloured by label and the
    edge row found by the scanner (dashed cyan line)."""
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    alpha = 0.35

    # Craters
    col_layer = np.zeros_like(overlay)
    col_layer[crater_mask] = C_CRATER_VIS
    overlay = cv2.addWeighted(overlay, 1 - alpha, col_layer, alpha, 0)

    # Crater contours + region numbers
    mask_u8 = crater_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, C_CRATER_VIS, 2)
    for i, cd in enumerate(crater_details, start=1):
        cy, cx = cd['centroid']
        cv2.putText(overlay, str(i), (int(cx) + 5, int(cy) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CRATER_VIS, 2)

    # Crack centrelines, coloured by label (crack k is label k of crack_labels)
    for crack in crack_info['individual_cracks']:
        colour = C_CRACK_LABEL_VIS.get(crack['label'], C_CRACK_VIS)
        for r, c in np.argwhere(crack_labels == crack['id']):
            cv2.circle(overlay, (int(c), int(r)), 1, colour, -1)

    # Edge row (dashed)
    if edge_end > 0:
        W = overlay.shape[1]
        for x0 in range(0, W, 16):
            cv2.line(overlay, (x0, int(edge_end)), (min(x0 + 9, W - 1), int(edge_end)),
                     C_EDGE_ROW_VIS, 1)
    return overlay


def fig_original_vs_segmentation(img_gray, overlay, crater_details, crack_info,
                                  size_cat, scale, edge_end, out_path):
    """Original SEM | segmentation overlay, with the measurements below.

    `scale` is the dict returned by `resolve_scale`; when the pixel size is not verified
    only pixel measurements are shown."""
    from matplotlib import gridspec

    plt.rcParams.update({'font.family': 'serif', 'font.size': 18})

    fig = plt.figure(figsize=(18, 12), facecolor='white')
    gs = gridspec.GridSpec(
        2, 2,
        height_ratios=[2.6, 2.0],
        hspace=0.0, wspace=0.06,
        left=0.03, right=0.97, top=0.97, bottom=0.02,
    )

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_seg  = fig.add_subplot(gs[0, 1])
    ax_info = fig.add_subplot(gs[1, :])   # single spanning info box

    # ── Image axes styling ─────────────────────────────────────────────────
    for ax in [ax_orig, ax_seg]:
        ax.set_facecolor('white')
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color('#AAAAAA')
            sp.set_linewidth(0.8)

    ax_orig.imshow(img_gray, cmap='gray')
    ax_orig.set_title('Original SEM Image', color='black',
                      fontsize=34, fontweight='bold', pad=12)

    ax_seg.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax_seg.set_title('U-Net Segmentation', color='black',
                     fontsize=34, fontweight='bold', pad=12)

    from matplotlib.lines import Line2D
    leg = [
        mpatches.Patch(color=(0.9, 0.63, 0.0), label=f'Crater like regions ({len(crater_details)})'),
        mpatches.Patch(color=(0.82, 0.0, 0.0), label=f'Edge connected cracks ({crack_info["n_edge_connected"]})'),
        mpatches.Patch(color=(0.82, 0.0, 0.82), label=f'Near edge cracks ({crack_info["n_near_edge"]})'),
        mpatches.Patch(color=(0.9, 0.82, 0.0), label=f'Internal cracks ({crack_info["n_internal"]})'),
        Line2D([0], [0], color=(0.0, 0.78, 1.0), ls='--', lw=2,
               label='Edge row' if edge_end > 0 else 'Edge row not found'),
    ]
    ax_seg.legend(handles=leg, loc='lower right', fontsize=23,
                  facecolor='white', labelcolor='black',
                  edgecolor='#BBBBBB', framealpha=0.95,
                  handlelength=2.0, handleheight=1.4, borderpad=0.9)

    # ── Single rounded info box ────────────────────────────────────────────
    ax_info.set_xlim(0, 1); ax_info.set_ylim(0, 1)
    ax_info.set_xticks([]); ax_info.set_yticks([])
    ax_info.set_facecolor('none')
    for sp in ax_info.spines.values():
        sp.set_visible(False)

    from matplotlib.patches import FancyBboxPatch
    ax_info.add_patch(FancyBboxPatch(
        (0.005, 0.04), 0.990, 0.92,
        boxstyle='round,pad=0.02',
        facecolor='#EEF2FF', edgecolor='#8899CC',
        linewidth=2.0, transform=ax_info.transAxes, clip_on=False,
    ))
    # Vertical divider
    ax_info.plot([0.5, 0.5], [0.10, 0.90], color='#8899CC',
                 linewidth=1.4, transform=ax_info.transAxes)

    # ── Build text content ─────────────────────────────────────────────────
    ci = crack_info
    n_c = len(crater_details)
    um = scale['um_per_px']
    cv_txt = (f"{ci['regular_cv']:.3f} ({ci['n_for_regularity']} near edge cracks)"
              if ci.get('regular_cv') is not None else
              f"not evaluated ({ci['n_for_regularity']} near edge cracks, 3 needed)")
    if um is not None:
        tot_area = sum(d['area_um2'] for d in crater_details)
        big_area = max((d['area_um2'] for d in crater_details), default=0.0)
        big_diam = max((d['diam_um'] for d in crater_details), default=0.0)
        txt_craters = (
            f"CRATER LIKE REGIONS\n"
            f"Pixel size: {um:.4f} µm/px ({scale['source']})\n"
            f"Regions: {n_c}  |  total area: {tot_area:.0f} µm²\n"
            f"Largest: {big_area:.0f} µm², eq. diam. {big_diam:.1f} µm\n"
            f"Size category: {SIZE_CATEGORIES[size_cat]}\n"
            f"(size class only, not a severity grade)"
        )
        txt_cracks = (
            f"CRACKS\n"
            f"Total: {ci['n_total']} (edge {ci['n_edge_connected']}, near {ci['n_near_edge']}, "
            f"int. {ci['n_internal']})\n"
            f"Total length: {ci['total_length_um']:.1f} µm\n"
            f"Longest: {ci['longest_um']:.1f} µm\n"
            f"Spacing CV: {cv_txt}"
        )
    else:
        tot_area = sum(d['area_px'] for d in crater_details)
        txt_craters = (
            f"CRATER LIKE REGIONS\n"
            f"PIXEL SIZE NOT VERIFIED:\n{scale['reason']}\n"
            f"Physical outputs withheld\n"
            f"Regions: {n_c}  |  total area: {tot_area} px"
        )
        txt_cracks = (
            f"CRACKS\n"
            f"Total: {ci['n_total']} (edge {ci['n_edge_connected']}, near {ci['n_near_edge']}, "
            f"int. {ci['n_internal']})\n"
            f"Total length: {ci['total_length_px']:.1f} px\n"
            f"Longest: {ci['longest_px']:.1f} px\n"
            f"Spacing CV: {cv_txt}"
        )

    ax_info.text(0.25, 0.52, txt_craters, transform=ax_info.transAxes,
                 ha='center', va='center', fontsize=22,
                 color='black', fontfamily='monospace', linespacing=1.75)
    ax_info.text(0.75, 0.52, txt_cracks, transform=ax_info.transAxes,
                 ha='center', va='center', fontsize=22,
                 color='black', fontfamily='monospace', linespacing=1.75)

    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', pad_inches=0.2,
                facecolor='white', edgecolor='none')
    plt.close(fig)


def fig_gt_comparison(img_gray, gt_crater, gt_crack, pred_crater, pred_crack_skel,
                       scores, out_path, gt_crack_thin=None):
    """Manual annotation vs prediction.

    Craters are scored by the strict IoU of the masks.  Cracks are scored by the
    length-weighted centreline F1 within 5 px (revision 1); the first version printed
    here a "crack IoU" between the 15-px annotation band and the predicted skeleton
    dilated by 4 px, which is not the protocol of the paper.
    """
    iou_c, f1_k = scores['crater_iou'], scores['crack_centreline_f1_5px']
    plt.rcParams.update({'font.family': 'serif', 'font.size': 17})

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.patch.set_facecolor('white')

    for row in axes:
        for ax in row:
            ax.set_facecolor('white')
            ax.tick_params(colors='black', labelsize=12)
            for sp in ax.spines.values():
                sp.set_color('#AAAAAA')
                sp.set_linewidth(0.7)
            ax.set_xticks([]); ax.set_yticks([])

    stem = out_path.stem.replace('_gt_comparison', '')
    fmt = lambda v: 'n/a' if v is None or v != v else f'{v:.3f}'   # noqa: E731
    fig.suptitle(
        f'{stem}\nCrater IoU (strict): {fmt(iou_c)}   |   '
        f'Crack centreline F1 (5 px): {fmt(f1_k)}',
        color='black', fontsize=30, fontweight='bold', y=0.99)

    # Annotated crater like regions
    gt_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    gt_vis[gt_crater > 0] = [255, 140, 0]
    axes[0, 0].imshow(gt_vis)
    axes[0, 0].set_title('Manual annotation: crater like regions', color='#CC5500',
                         fontsize=26, fontweight='bold')

    # Predicted crater like regions
    pred_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    pred_vis[pred_crater] = [255, 140, 0]
    axes[0, 1].imshow(pred_vis)
    axes[0, 1].set_title(f'Prediction: crater like regions\n(IoU, strict: {fmt(iou_c)})',
                         color='#CC5500', fontsize=26, fontweight='bold')

    # Annotated crack centrelines (thin polylines when available)
    gt_vis_k = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    gt_crack_display = gt_crack_thin if gt_crack_thin is not None else skeletonize(gt_crack.astype(bool))
    gt_vis_k[binary_dilation(gt_crack_display, disk(1))] = [210, 0, 0]
    axes[1, 0].imshow(gt_vis_k)
    axes[1, 0].set_title('Manual annotation: crack centrelines', color='#AA0000',
                         fontsize=26, fontweight='bold')

    # Predicted crack centrelines
    pred_vis_k = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    skel_dilated = binary_dilation(pred_crack_skel, disk(1))
    pred_vis_k[skel_dilated] = [210, 0, 0]
    axes[1, 1].imshow(pred_vis_k)
    axes[1, 1].set_title(f'Prediction: crack centrelines\n(centreline F1 within 5 px: {fmt(f1_k)})',
                         color='#AA0000', fontsize=26, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.87])
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', pad_inches=0.15,
                facecolor='white', edgecolor='none')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def resolve_scale(gray_full, user_um_per_px=None):
    """Pixel size of an image and its provenance.

    Order: the value given by the user (source 'user'), otherwise the data bar when the
    printed magnification and field width agree within 1 % (source 'data bar').  When
    neither is available `um_per_px` is None and every physical output is withheld.
    """
    if user_um_per_px is not None:
        return {'um_per_px': float(user_um_per_px), 'verified': False, 'source': 'user',
                'magnification': None, 'width_um': None,
                'reason': 'pixel size given by the user (--scale)'}
    res = read_scale(gray_full)
    if res['um_per_px'] is None:
        res['source'] = None
    return res


def analysis_rows(gray_full, strip_fraction=STRIP_FRACTION):
    """Rows analysed: the training crop (frame without its bottom `strip_fraction` of
    rows), further limited to the rows above the data bar."""
    H = gray_full.shape[0]
    roi_h = H - int(H * strip_fraction)
    bar_top = MS.locate_data_bar(gray_full)
    return roi_h, min(roi_h, bar_top), bar_top


def _strict_iou(ref, pred):
    ref, pred = ref.astype(bool), pred.astype(bool)
    union = (ref | pred).sum()
    return float((ref & pred).sum() / union) if union else float('nan')


def analyze_image(predictor, img_path, output_dir, gt_json_path=None,
                  strip_fraction=STRIP_FRACTION, user_um_per_px=None):
    img_path = Path(img_path)
    output_dir = Path(output_dir)
    stem = img_path.stem
    print(f"\n  {'='*60}\n  {stem}\n  {'='*60}")

    # 1. Load image
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read: {img_path}")
    img_gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H_full, W_full = img_gray_full.shape

    # 2. Region of interest: training crop, then rows above the data bar
    roi_h, valid_h, bar_top = analysis_rows(img_gray_full, strip_fraction)
    roi_net = img_gray_full[:roi_h, :]
    img_gray = img_gray_full[:valid_h, :]
    H, W = img_gray.shape
    print(f"  Image: {W_full}x{H_full} px | network input: rows 0-{roi_h} "
          f"| analysed rows: 0-{valid_h} (data bar from row {bar_top})")

    # 3. Pixel size (verified, given by the user, or withheld)
    scale = resolve_scale(img_gray_full, user_um_per_px)
    um_per_px = scale['um_per_px']
    flags = []
    if um_per_px is None:
        flags.append('scale not verified: physical outputs withheld')
        print(f"  [Scale] NOT VERIFIED ({scale['reason']}). Physical outputs withheld; "
              f"give the pixel size with --scale to obtain them.")
    else:
        print(f"  Scale: {um_per_px:.4f} µm/px [{scale['source']}] "
              f"({W * um_per_px:.0f} x {H * um_per_px:.0f} µm analysed)")

    # 4. Edge row
    edge_end, artifact_start = compute_zones_dynamic(img_gray, bottom_pct=0.02)
    if edge_end == 0:
        flags.append('edge row not found: every crack is labelled internal or near edge')
    print(f"  Edge row: {edge_end} px" if edge_end else "  Edge row: not found")

    # 5. U-Net prediction on the training crop, restricted to the analysed rows
    print(f"\n  [U-Net] Running inference (TTA)...")
    crater_mask, crack_mask, _, _ = predictor.predict(roi_net)
    crater_mask, crack_mask = crater_mask[:valid_h], crack_mask[:valid_h]

    # 6. Analysis stage
    crater_details, size_cat, crater_labels = analyze_craters(
        crater_mask, img_gray, um_per_px, edge_end, artifact_start)
    kept = (np.isin(crater_labels, [d['label'] for d in crater_details])
            if crater_details else np.zeros_like(crater_mask, dtype=bool))
    crack_info, crack_skel, crack_labels = analyze_cracks(crack_mask, um_per_px, edge_end)

    # 7. Print results
    ci = crack_info
    print(f"\n  ── CRATER LIKE REGIONS ──")
    print(f"    Count: {len(crater_details)}")
    if um_per_px is not None:
        if crater_details:
            big = max(crater_details, key=lambda x: x['area_um2'])
            print(f"    Total area: {sum(d['area_um2'] for d in crater_details):.1f} µm² | "
                  f"largest: {big['area_um2']:.1f} µm², eq. diameter {big['diam_um']:.1f} µm")
        print(f"    Size category of the largest region: {SIZE_CATEGORIES[size_cat]} "
              f"(size class only, not a severity grade)")
    print(f"\n  ── CRACKS ──")
    print(f"    Total: {ci['n_total']} (edge connected {ci['n_edge_connected']}, "
          f"near edge {ci['n_near_edge']}, internal {ci['n_internal']})")
    if um_per_px is not None:
        print(f"    Total length: {ci['total_length_um']:.1f} µm | Longest: {ci['longest_um']:.1f} µm")
    else:
        print(f"    Total length: {ci['total_length_px']:.1f} px | Longest: {ci['longest_px']:.1f} px")
    print(f"    Spacing CV: {ci['regular_cv'] if ci['regular_cv'] is not None else 'not evaluated'}"
          f" ({ci['n_for_regularity']} edge connected or near edge cracks)")

    # 8. Segmentation figure
    overlay = build_segmentation_overlay(img_gray, kept, crack_labels, crack_info, edge_end,
                                         crater_details)
    seg_path = output_dir / f"{stem}_segmentation.png"
    fig_original_vs_segmentation(img_gray, overlay, crater_details, crack_info,
                                 size_cat, scale, edge_end, seg_path)
    print(f"\n  -> {seg_path.name}")

    # 9. Comparison with a manual annotation (if available)
    gt_metrics = {}
    if gt_json_path is None:
        for search_dir in [img_path.parent.parent / 'masks',
                           img_path.parent.parent / 'data' / 'masks',
                           img_path.parent]:
            candidate = search_dir / f"{stem}.json"
            if candidate.exists():
                gt_json_path = candidate
                break

    if gt_json_path and Path(gt_json_path).exists():
        gt_crater, gt_crack = load_gt_masks(gt_json_path, H, W, H_full, W_full)
        gt_crack_thin = load_gt_crack_thin(gt_json_path, H, W, H_full, W_full)
        cl = MS.centreline_scores(gt_crack_thin, crack_mask, tau=5.0)
        gt_metrics = {
            'crater_iou': round(_strict_iou(gt_crater, crater_mask), 4),
            'crack_centreline_precision_5px': round(cl['precision'], 4),
            'crack_centreline_recall_5px': round(cl['recall'], 4),
            'crack_centreline_f1_5px': round(cl['f1'], 4),
            'note': 'raw masks of the deployed pipeline, analysed rows only; '
                    'crater IoU strict, crack scores length weighted within 5 px',
        }
        gt_path = output_dir / f"{stem}_gt_comparison.png"
        fig_gt_comparison(img_gray, gt_crater, gt_crack, crater_mask, crack_skel,
                          gt_metrics, gt_path, gt_crack_thin=gt_crack_thin)
        print(f"  -> {gt_path.name}  (crater IoU {gt_metrics['crater_iou']:.3f}, "
              f"crack centreline F1 {gt_metrics['crack_centreline_f1_5px']:.3f})")

    # 10. JSON report
    scaled = um_per_px is not None
    crack_summary = {k: v for k, v in crack_info.items() if k != 'individual_cracks'}
    report = {
        'file': img_path.name,
        'software': 'tungsten_analyzer.inference, revision 1',
        'dimensions': {'width_px': W_full, 'height_px': H_full},
        'flags': flags,
        'scale': {
            'um_per_px': um_per_px,
            'source': scale['source'],
            'verified': scale['verified'],
            'magnification_read': scale['magnification'],
            'field_width_um_read': scale['width_um'],
            'status': scale['reason'],
            'analysed_field_um': f"{W * um_per_px:.1f} x {H * um_per_px:.1f}" if scaled else None,
        },
        'rows': {
            'network_input_rows': roi_h,
            'data_bar_top_row': bar_top,
            'analysed_rows': valid_h,
        },
        'edge_row_px': edge_end if edge_end > 0 else None,
        'craters': {
            'count': len(crater_details),
            'total_area_um2': round(sum(d['area_um2'] for d in crater_details), 1) if scaled else None,
            'largest_area_um2': round(max((d['area_um2'] for d in crater_details), default=0.0), 1) if scaled else None,
            'largest_diameter_um': round(max((d['diam_um'] for d in crater_details), default=0.0), 1) if scaled else None,
            'total_area_px': int(sum(d['area_px'] for d in crater_details)),
            'size_category': size_cat,
            'size_category_label': SIZE_CATEGORIES[size_cat] if size_cat is not None else None,
            'size_category_note': 'size class of the largest region only, '
                                  'not calibrated against any physical endpoint',
            'details': [{k: v for k, v in d.items() if k not in ('centroid', 'bbox')}
                        for d in crater_details],
        },
        'cracks': crack_summary,
        'crack_details': crack_info['individual_cracks'],
        'gt_comparison': gt_metrics if gt_metrics else None,
    }

    report_path = output_dir / f"{stem}_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"  -> {report_path.name}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
#  CLI & BATCH
# ─────────────────────────────────────────────────────────────────────────────

SUMMARY_FIELDS = [
    'file', 'flags', 'scale_um_px', 'scale_source', 'scale_status', 'edge_row_px',
    'analysed_rows', 'n_crater_regions', 'crater_total_area_um2', 'crater_largest_area_um2',
    'crater_largest_diam_um', 'size_category', 'size_category_label',
    'n_cracks', 'n_edge_connected', 'n_near_edge', 'n_internal',
    'crack_total_length_um', 'crack_longest_um', 'crack_total_length_px',
    'regular_cv', 'n_cracks_for_cv',
    'gt_crater_iou', 'gt_crack_centreline_f1_5px', 'error',
]


def summary_row(report):
    sc, cr, ck = report['scale'], report['craters'], report['cracks']
    gt = report.get('gt_comparison') or {}
    return {
        'file': report['file'],
        'flags': '; '.join(report['flags']),
        'scale_um_px': sc['um_per_px'],
        'scale_source': sc['source'],
        'scale_status': sc['status'],
        'edge_row_px': report['edge_row_px'],
        'analysed_rows': report['rows']['analysed_rows'],
        'n_crater_regions': cr['count'],
        'crater_total_area_um2': cr['total_area_um2'],
        'crater_largest_area_um2': cr['largest_area_um2'],
        'crater_largest_diam_um': cr['largest_diameter_um'],
        'size_category': cr['size_category'],
        'size_category_label': cr['size_category_label'],
        'n_cracks': ck['n_total'],
        'n_edge_connected': ck['n_edge_connected'],
        'n_near_edge': ck['n_near_edge'],
        'n_internal': ck['n_internal'],
        'crack_total_length_um': ck['total_length_um'],
        'crack_longest_um': ck['longest_um'],
        'crack_total_length_px': ck['total_length_px'],
        'regular_cv': ck['regular_cv'],
        'n_cracks_for_cv': ck['n_for_regularity'],
        'gt_crater_iou': gt.get('crater_iou', ''),
        'gt_crack_centreline_f1_5px': gt.get('crack_centreline_f1_5px', ''),
    }


def main():
    p = argparse.ArgumentParser(description='U-Net Full Inference Pipeline for Tungsten SEM')
    p.add_argument('--input', default=None, help='Image path or directory (with --batch)')
    p.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    p.add_argument('--output-dir', default='./inference_results')
    p.add_argument('--gt', default=None, help='Path to the annotation JSON (optional)')
    p.add_argument('--batch', action='store_true', help='Process all images in directory')
    p.add_argument('--strip-fraction', type=float, default=STRIP_FRACTION,
                   help='Bottom fraction of rows removed, as in training (default 0.08)')
    p.add_argument('--scale', type=float, default=None,
                   help='Pixel size in µm/px.  Overrides the data bar and is recorded in '
                        'the report with source "user".  Without it, the physical outputs '
                        'of an image whose data bar cannot be verified are withheld.')
    args = p.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  {'═'*60}")
    print(f"  TUNGSTEN SEM ANALYZER — U-Net Inference Pipeline (revision 1)")
    print(f"  {'═'*60}")

    # Load model
    predictor = UNetPredictor(args.checkpoint)

    # Interactive mode if no input
    if args.input is None:
        args.input = input("\n  Enter image path or folder:\n  > ").strip('"\'')

    input_path = Path(args.input)
    EXTS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

    if args.batch or input_path.is_dir():
        imgs = sorted([p for p in input_path.iterdir() if p.suffix.lower() in EXTS])
        print(f"\n  Batch mode: {len(imgs)} images found")

        rows = []
        for img_path in imgs:
            try:
                report = analyze_image(predictor, img_path, output_dir,
                                       strip_fraction=args.strip_fraction,
                                       user_um_per_px=args.scale)
                rows.append(summary_row(report))
            except Exception as e:
                print(f"  ERROR: {img_path.name}: {e}")
                rows.append({'file': img_path.name, 'error': str(e)})

        # Save CSV summary
        if rows:
            csv_path = output_dir / 'summary.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction='ignore')
                w.writeheader()
                w.writerows(rows)
            n_flag = sum(1 for r in rows if r.get('flags') or r.get('error'))
            print(f"\n  >>> Summary CSV: {csv_path}  ({n_flag} of {len(rows)} images flagged)")
    else:
        analyze_image(predictor, input_path, output_dir, gt_json_path=args.gt,
                      strip_fraction=args.strip_fraction, user_um_per_px=args.scale)

    print(f"\n  {'═'*60}")
    print(f"  All results saved to: {output_dir}/")
    print(f"  {'═'*60}\n")


if __name__ == '__main__':
    main()
