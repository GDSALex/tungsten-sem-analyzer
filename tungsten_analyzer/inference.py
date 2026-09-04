"""
=============================================================================
UNET_FULL_INFERENCE.PY  —  Complete Inference Pipeline
=============================================================================
Combines U-Net segmentation with classical post-processing to produce
publication-ready analysis of tungsten SEM images.

PIPELINE:
  1. Load image + auto-scale (OCR from SEM metadata bar)
  2. U-Net prediction (craters + cracks)
  3. Post-processing: skeletonization, artifact filtering, zone detection
  4. Classification: crater grades (0-4), crack types (large/small/regular)
  5. Metrics: areas, lengths, diameters in µm
  6. Figures: original vs segmentation, GT comparison, detailed report

USAGE:
  python unet_full_inference.py                           # Interactive mode
  python unet_full_inference.py --input image.jpg         # Single image
  python unet_full_inference.py --input ./folder --batch  # Batch mode
  python unet_full_inference.py --input image.jpg --gt mask.json  # With GT comparison

OUTPUT (per image):
  - {name}_segmentation.png    : Original vs U-Net segmentation
  - {name}_gt_comparison.png   : GT vs prediction (if JSON provided)
  - {name}_report.json         : Full quantitative report
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

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

CRATER_GRADES = {
    0: (0,     0,    "No structural damage"),
    1: (1,     100,  "Micro thermal damage (< 100 µm)"),
    2: (100,   600,  "Melting and crumbling (100–600 µm)"),
    3: (600,   950,  "Vaporization crater (600–950 µm)"),
    4: (950, 99999,  "Catastrophic damage (> 950 µm)"),
}

CRACK_CLASSES = {
    'large': {0: 'None', 1: '< 1000 µm', 2: '≥ 1000 µm'},
    'small': {0: 'None', 1: '1–2 cracks', 2: '> 2 cracks'},
    'regular': {0: 'No', 1: 'Yes (CV < 45%)'},
}

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# JSON mask labels
_CRATER_LABELS = {'crater', 'cratera', 'cráter', 'vaporization',
                  'collapse', 'crater_vaporization', 'crater_collapse'}
_CRACK_LABELS  = {'crack', 'grieta', 'fracture', 'fractura',
                  'crack_large', 'crack_small'}


# ─────────────────────────────────────────────────────────────────────────────
#  CLASSICAL INFRASTRUCTURE (from Tungsten_Analyzer.py)
# ─────────────────────────────────────────────────────────────────────────────

def detect_meta_strip(gray):
    h = gray.shape[0]
    for frac in (0.15, 0.12, 0.10, 0.08):
        s = int(h * frac)
        strip = gray[h - s:, :]
        if strip.mean() < 110:
            _, bw = cv2.threshold(strip, 190, 255, cv2.THRESH_BINARY)
            if bw.sum() / 255 > 30:
                return s
    return int(h * 0.10)


def auto_scale(gray, meta_h):
    try:
        import pytesseract
        from PIL import Image
        import re
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        strip = gray[-meta_h:, :]
        _, thresh = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        h_s, w_s = thresh.shape
        thresh = cv2.resize(thresh, (w_s * 3, h_s * 3), interpolation=cv2.INTER_CUBIC)
        pil = Image.fromarray(thresh)
        text = pytesseract.image_to_string(pil, config='--psm 6').replace('\n', ' ')
        w_px = gray.shape[1]
        m_mag = re.search(r'[Mm]ag\s*[=:]?\s*(\d+)', text)
        if m_mag:
            mag_val = float(m_mag.group(1))
            sc = 111.8 / mag_val if mag_val > 0 else None
            if sc:
                print(f"  [OCR] Magnification: {mag_val:.0f}x -> {sc:.4f} µm/px")
                return sc
        for pat, mult in [(r'[Ww]idth\s*[=:]\s*([\d.]+)\s*mm', 1000.0),
                          (r'[Ww]idth\s*[=:]\s*([\d.]+)\s*[µu]m', 1.0)]:
            m = re.search(pat, text)
            if m:
                um = float(m.group(1)) * mult
                sc = um / w_px
                print(f"  [OCR] Physical width: {um:.1f} µm -> {sc:.4f} µm/px")
                return sc
    except Exception as e:
        print(f"  [OCR] Tesseract failed: {e}")
    return None


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
    Only pixels that bridge between two existing crack regions are kept,
    avoiding spurious expansion.
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
            # Verify bridging: dilate original mask slightly and check
            # that new pixels are within reach of existing crack on both ends
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
    """Analyze crater mask: label, measure, classify, filter artifacts."""
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

        diam_px = getattr(pp, 'equivalent_diameter_area',
                          getattr(pp, 'equivalent_diameter', 0.0))
        diam_um = float(diam_px) * um_per_px
        area_um2 = area_px * (um_per_px ** 2)
        perimeter_um = pp.perimeter * um_per_px

        # Grade assignment
        if diam_um < 1:      grade = 0
        elif diam_um < 100:  grade = 1
        elif diam_um < 600:  grade = 2
        elif diam_um < 950:  grade = 3
        else:                grade = 4

        details.append({
            'label': pp.label,
            'area_px': area_px,
            'area_um2': round(area_um2, 1),
            'diam_um': round(diam_um, 1),
            'perimeter_um': round(perimeter_um, 1),
            'centroid': (float(pp.centroid[0]), float(pp.centroid[1])),
            'bbox': tuple(int(v) for v in pp.bbox),
            'eccentricity': round(float(pp.eccentricity), 3),
            'solidity': round(float(pp.solidity), 3),
            'grade': grade,
            'grade_desc': CRATER_GRADES[grade][2],
        })

    max_grade = max((d['grade'] for d in details), default=0)
    return details, max_grade, labels


def analyze_cracks(crack_mask, um_per_px, edge_end):
    """Skeletonize crack mask and compute per-crack metrics."""
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
        'large_crack_class_desc': CRACK_CLASSES['large'][large_cls],
        'large_crack_max_um': round(large_max_um, 1),
        'small_crack_class': small_cls,
        'small_crack_class_desc': CRACK_CLASSES['small'][small_cls],
        'regular_crack_class': regular_cls,
        'regular_crack_class_desc': CRACK_CLASSES['regular'][regular_cls],
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


def build_segmentation_overlay(img_gray, crater_mask, crack_skel,
                                crater_details, edge_end):
    overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    alpha = 0.35

    # Craters
    col_layer = np.zeros_like(overlay)
    col_layer[crater_mask] = C_CRATER_VIS
    overlay = cv2.addWeighted(overlay, 1 - alpha, col_layer, alpha, 0)
    
    # Crater contours + labels
    mask_u8 = crater_mask.astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, C_CRATER_VIS, 2)
    
    for cd in crater_details:
        cy, cx = cd['centroid']
        cv2.putText(overlay, f"G{cd['grade']}",
                    (int(cx)+5, int(cy)-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CRATER_VIS, 2)
    
    # Crack skeleton
    for r, c in np.argwhere(crack_skel):
        cv2.circle(overlay, (c, r), 1, C_CRACK_VIS, -1)
    
    return overlay


def fig_original_vs_segmentation(img_gray, overlay, crater_details, crack_info,
                                  max_grade, um_per_px, edge_end, out_path):
    """Publication-ready figure: Original SEM | Segmentation overlay."""
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

    leg = [
        mpatches.Patch(color=(0.9, 0.63, 0.0), label=f'Craters ({len(crater_details)})'),
        mpatches.Patch(color=(0.82, 0.0, 0.0),  label=f'Cracks ({crack_info["n_total"]})'),
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
    gd = CRATER_GRADES.get(max_grade, (0, 0, ''))[2]
    n_c = len(crater_details)
    max_crater_um   = max((d['diam_um']  for d in crater_details), default=0)
    max_crater_area = max((d['area_um2'] for d in crater_details), default=0)

    txt_craters = (
        f"CRATERS\n"
        f"Scale: {um_per_px:.4f} µm/px\n"
        f"Count: {n_c}  |  Grade: {max_grade}\n"
        f"{gd}\n"
        f"Largest diam.: {max_crater_um:.1f} µm\n"
        f"Largest area: {max_crater_area:.1f} µm²"
    )
    txt_cracks = (
        f"CRACKS\n"
        f"Total: {ci['n_total']} "
        f"(edge: {ci['n_large']}, int: {ci['n_small']})\n"
        f"Total length: {ci['total_length_um']:.1f} µm\n"
        f"Longest: {ci['max_single_crack_um']:.1f} µm\n"
        f"Large: {ci['large_crack_class']}  |  Small: {ci['small_crack_class']}\n"
        f"Regular: {ci['regular_crack_class_desc']}"
        + (f" (CV={ci['regular_cv']:.3f})" if ci.get('regular_cv') is not None else "")
    )

    ax_info.text(0.25, 0.52, txt_craters, transform=ax_info.transAxes,
                 ha='center', va='center', fontsize=28,
                 color='black', fontfamily='monospace', linespacing=1.75)
    ax_info.text(0.75, 0.52, txt_cracks, transform=ax_info.transAxes,
                 ha='center', va='center', fontsize=28,
                 color='black', fontfamily='monospace', linespacing=1.75)

    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', pad_inches=0.2,
                facecolor='white', edgecolor='none')
    plt.close(fig)


def fig_gt_comparison(img_gray, gt_crater, gt_crack, pred_crater, pred_crack_skel,
                       iou_c, iou_k, out_path, gt_crack_thin=None):
    """Publication-ready figure: GT masks vs Prediction."""
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
    fig.suptitle(
        f'{stem}\nCrater IoU: {iou_c:.3f}   |   Crack IoU: {iou_k:.3f}',
        color='black', fontsize=34, fontweight='bold', y=0.99)

    # GT Craters
    gt_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    gt_vis[gt_crater > 0] = [255, 140, 0]
    axes[0, 0].imshow(gt_vis)
    axes[0, 0].set_title('Ground Truth — Craters', color='#CC5500', fontsize=30, fontweight='bold')

    # Pred Craters
    pred_vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    pred_vis[pred_crater] = [255, 140, 0]
    axes[0, 1].imshow(pred_vis)
    axes[0, 1].set_title(f'U-Net Prediction — Craters\n(IoU: {iou_c:.3f})',
                         color='#CC5500', fontsize=30, fontweight='bold')

    # GT Cracks — use thin linestrip mask when available (no branching artifacts)
    gt_vis_k = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    gt_crack_display = gt_crack_thin if gt_crack_thin is not None else skeletonize(gt_crack.astype(bool))
    gt_vis_k[binary_dilation(gt_crack_display, disk(1))] = [210, 0, 0]
    axes[1, 0].imshow(gt_vis_k)
    axes[1, 0].set_title('Ground Truth — Cracks', color='#AA0000', fontsize=30, fontweight='bold')

    # Pred Cracks (skeleton)
    pred_vis_k = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)
    skel_dilated = binary_dilation(pred_crack_skel, disk(1))
    pred_vis_k[skel_dilated] = [210, 0, 0]
    axes[1, 1].imshow(pred_vis_k)
    axes[1, 1].set_title(f'U-Net Prediction — Cracks\n(IoU: {iou_k:.3f})',
                         color='#AA0000', fontsize=30, fontweight='bold')

    plt.tight_layout(rect=[0, 0, 1, 0.87])
    plt.savefig(str(out_path), dpi=150, bbox_inches='tight', pad_inches=0.15,
                facecolor='white', edgecolor='none')
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def analyze_image(predictor, img_path, output_dir, gt_json_path=None,
                  strip_fraction=0.08):
    img_path = Path(img_path)
    stem = img_path.stem
    print(f"\n  {'='*60}\n  {stem}\n  {'='*60}")
    
    # 1. Load image
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise ValueError(f"Cannot read: {img_path}")
    img_gray_full = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    H_full, W_full = img_gray_full.shape
    
    # 2. Detect and strip metadata bar
    meta_h = detect_meta_strip(img_gray_full)
    img_gray = img_gray_full[:H_full - meta_h, :]
    H, W = img_gray.shape
    print(f"  Image: {W_full}x{H_full} px | Metadata strip: {meta_h} px | ROI: {W}x{H} px")
    
    # 3. Auto-scale
    um_per_px = auto_scale(img_gray_full, meta_h)
    if um_per_px is None:
        um_per_px = 0.224
        print(f"  [Scale] Auto-detection failed. Using default: {um_per_px} µm/px (500X)")
    print(f"  Scale: {um_per_px:.4f} µm/px ({W * um_per_px:.0f} x {H * um_per_px:.0f} µm)")
    
    # 4. Detect zones
    edge_end, artifact_start = compute_zones_dynamic(img_gray)
    print(f"  Edge zone: 0–{edge_end} px | Interior: {edge_end}–{artifact_start} px")
    
    # 5. U-Net prediction
    print(f"\n  [U-Net] Running inference (TTA)...")
    crater_mask, crack_mask, crater_prob, crack_prob = predictor.predict(img_gray)
    
    # 6. Post-processing
    print(f"  [Post] Analyzing craters...")
    crater_details, max_grade, crater_labels = analyze_craters(
        crater_mask, img_gray, um_per_px, edge_end, artifact_start)
    
    print(f"  [Post] Skeletonizing cracks...")
    crack_info, crack_skel, crack_labels = analyze_cracks(
        crack_mask, um_per_px, edge_end)
    
    # 7. Print results
    ci = crack_info
    gd = CRATER_GRADES.get(max_grade, (0, 0, ''))[2]
    print(f"\n  ── CRATERS ──")
    print(f"    Count: {len(crater_details)}  |  Max grade: {max_grade} — {gd}")
    if crater_details:
        max_d = max(crater_details, key=lambda x: x['diam_um'])
        print(f"    Largest: {max_d['diam_um']:.1f} µm diameter, {max_d['area_um2']:.1f} µm² area")
    print(f"\n  ── CRACKS ──")
    print(f"    Total: {ci['n_total']} (edge: {ci['n_large']}, internal: {ci['n_small']})")
    print(f"    Total length: {ci['total_length_um']:.1f} µm | Longest: {ci['max_single_crack_um']:.1f} µm")
    print(f"    Large class: {ci['large_crack_class']} ({ci['large_crack_class_desc']})")
    print(f"    Small class: {ci['small_crack_class']} ({ci['small_crack_class_desc']})")
    print(f"    Regular: {ci['regular_crack_class_desc']}")
    
    # 8. Generate segmentation figure
    overlay = build_segmentation_overlay(img_gray, crater_mask, crack_skel,
                                          crater_details, edge_end)
    seg_path = output_dir / f"{stem}_segmentation.png"
    fig_original_vs_segmentation(img_gray, overlay, crater_details, crack_info,
                                  max_grade, um_per_px, edge_end, seg_path)
    print(f"\n  -> {seg_path.name}")
    
    # 9. GT comparison (if available)
    gt_metrics = {}
    if gt_json_path is None:
        # Auto-search for GT JSON
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
        iou_c = compute_iou(gt_crater, crater_mask)
        iou_k = compute_iou(gt_crack, binary_dilation(crack_skel, disk(4)))
        gt_metrics = {'crater_iou': round(iou_c, 4), 'crack_iou': round(iou_k, 4)}

        gt_path = output_dir / f"{stem}_gt_comparison.png"
        fig_gt_comparison(img_gray, gt_crater, gt_crack, crater_mask, crack_skel,
                           iou_c, iou_k, gt_path, gt_crack_thin=gt_crack_thin)
        print(f"  -> {gt_path.name}  (Crater IoU: {iou_c:.3f}, Crack IoU: {iou_k:.3f})")
    
    # 10. Save JSON report
    report = {
        'file': img_path.name,
        'dimensions': {'width_px': W_full, 'height_px': H_full},
        'scale': {
            'um_per_px': um_per_px,
            'field_of_view_um': f"{W * um_per_px:.1f} x {H * um_per_px:.1f}",
        },
        'zones': {
            'edge_end_px': edge_end,
            'artifact_start_px': artifact_start,
        },
        'craters': {
            'count': len(crater_details),
            'max_grade': max_grade,
            'max_grade_desc': gd,
            'max_diameter_um': round(max((d['diam_um'] for d in crater_details), default=0), 1),
            'max_area_um2': round(max((d['area_um2'] for d in crater_details), default=0), 1),
            'total_area_um2': round(sum(d['area_um2'] for d in crater_details), 1),
            'details': [{k: v for k, v in d.items() if k != 'centroid' and k != 'bbox'}
                        for d in crater_details],
        },
        'cracks': {k: v for k, v in crack_info.items() if k != 'individual_cracks'},
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

def main():
    p = argparse.ArgumentParser(description='U-Net Full Inference Pipeline for Tungsten SEM')
    p.add_argument('--input', default=None, help='Image path or directory (with --batch)')
    p.add_argument('--checkpoint', default='checkpoints/best_model.pth')
    p.add_argument('--output-dir', default='./inference_results')
    p.add_argument('--gt', default=None, help='Path to GT JSON (optional)')
    p.add_argument('--batch', action='store_true', help='Process all images in directory')
    p.add_argument('--strip-fraction', type=float, default=0.08)
    args = p.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n  {'═'*60}")
    print(f"  TUNGSTEN SEM ANALYZER — U-Net Inference Pipeline")
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
                                        strip_fraction=args.strip_fraction)
                rows.append({
                    'file': report['file'],
                    'scale_um_px': report['scale']['um_per_px'],
                    'n_craters': report['craters']['count'],
                    'max_crater_grade': report['craters']['max_grade'],
                    'max_crater_diam_um': report['craters']['max_diameter_um'],
                    'max_crater_area_um2': report['craters']['max_area_um2'],
                    'total_crater_area_um2': report['craters']['total_area_um2'],
                    'n_cracks': report['cracks']['n_total'],
                    'n_large_cracks': report['cracks']['n_large'],
                    'n_small_cracks': report['cracks']['n_small'],
                    'total_crack_length_um': report['cracks']['total_length_um'],
                    'max_crack_length_um': report['cracks']['max_single_crack_um'],
                    'large_crack_class': report['cracks']['large_crack_class'],
                    'small_crack_class': report['cracks']['small_crack_class'],
                    'regular_crack_class': report['cracks']['regular_crack_class'],
                    'regular_cv': report['cracks'].get('regular_cv', ''),
                    'crater_iou': report.get('gt_comparison', {}).get('crater_iou', ''),
                    'crack_iou': report.get('gt_comparison', {}).get('crack_iou', ''),
                })
            except Exception as e:
                print(f"  ERROR: {img_path.name}: {e}")
                rows.append({'file': img_path.name, 'error': str(e)})

        # Save CSV summary
        if rows:
            fieldnames = [
                'file', 'scale_um_px',
                'n_craters', 'max_crater_grade', 'max_crater_diam_um',
                'max_crater_area_um2', 'total_crater_area_um2',
                'n_cracks', 'n_large_cracks', 'n_small_cracks',
                'total_crack_length_um', 'max_crack_length_um',
                'large_crack_class', 'small_crack_class', 'regular_crack_class', 'regular_cv',
                'crater_iou', 'crack_iou', 'error',
            ]
            csv_path = output_dir / 'summary.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
                w.writeheader()
                w.writerows(rows)
            print(f"\n  >>> Summary CSV: {csv_path}")
    else:
        analyze_image(predictor, input_path, output_dir, gt_json_path=args.gt,
                       strip_fraction=args.strip_fraction)
    
    print(f"\n  {'═'*60}")
    print(f"  All results saved to: {output_dir}/")
    print(f"  {'═'*60}\n")


if __name__ == '__main__':
    main()