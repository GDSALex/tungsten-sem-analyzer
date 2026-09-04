"""Classical computer vision pipeline for tungsten SEM micrographs.

This is the first of the three stages described in the paper and it predates the
network. It was written when no annotated dataset existed yet, which is exactly what
made it useful: its output, once tuned, became the pre-annotation that turned
labelling a hundred micrographs from a prohibitive job into a feasible one.

The pipeline has four parts.

A dynamic topographic scanner locates the row at which the material surface begins,
from the texture energy profile of the frame rather than from raw intensity, so that
it is indifferent to whether the background above the edge is bright, dark or noisy.
Everything above that row is excluded from the analysis. Without this step the
crater and crack detectors generate thousands of detections in a region that carries
no information.

Crater-like damage is segmented with a multichannel watershed. Damaged regions do
not share a single appearance in SEM, they may be dark depressions, bright ablated
areas, roughened zones or collapsed surface, so several complementary feature maps
are computed and used jointly to seed the watershed. Candidate regions are then
filtered on morphology, solidity, eccentricity, contour irregularity and equivalent
diameter, rather than on grey level, which is what makes the detector survive
changes of contrast between images.

Cracks are enhanced with a multiscale Frangi filter combined with a morphological
black-hat transform, and binarised by dual hysteresis, permissive on sound material
and strict inside craters where ablation roughness would otherwise be read as a
tangle of false cracks. A topological filter then removes short isolated fragments.

Finally each crater is graded on a five-level severity scale and each crack is
classified as edge-connected or internal, with a regularity metric computed from the
coefficient of variation of the inter-crack spacing.

The forty parameters in ``CFG`` are the configuration found by the evolutionary
search of :mod:`tungsten_analyzer.genetic_search`, and they are the ones behind
every classical-pipeline result reported in the paper. The performance ceiling of
this approach, in particular its inability to resolve cracks narrower than about
four or five pixels without also firing on grain boundaries, is what motivated the
move to deep learning.

The edge scanner remains in use inside the deep-learning workflow, where it supplies
the physical frame of reference in which the network output is interpreted.

Command line usage::

    python -m tungsten_analyzer.classical_pipeline image.jpg
    python -m tungsten_analyzer.classical_pipeline image.jpg --scale 0.224
    python -m tungsten_analyzer.classical_pipeline ./images/ --batch

Arguments::

    --scale FLOAT         um per pixel. When omitted the scale is read from the SEM
                          metadata bar by OCR.
    --output-dir PATH     where figures and JSON reports are written
    --bottom-pct FLOAT    fraction of the frame treated as the artefact band (0.05)
    --crack-thresh FLOAT  crack sensitivity (0.18). Lower it to catch micro-cracks,
                          raise it to suppress grain boundaries.
    --min-crack-um FLOAT  minimum length in um for a crack to count (15)
    --batch               process a whole directory and write a summary CSV

External dependencies: numpy, opencv-python, scipy, scikit-image, matplotlib and
pytesseract. The last one needs the Tesseract OCR binary installed on the system;
without it the scale must be given with --scale.
"""

import argparse
import json
import csv
import re
import warnings
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage
from skimage.morphology import (skeletonize, remove_small_objects,
                                 binary_dilation, disk)
from skimage.filters import frangi
from skimage.measure import regionprops, label as ski_label
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import cv2
from skimage.measure import label as ski_label
from skimage.morphology import remove_small_objects

warnings.filterwarnings('ignore')


CRATER_GRADES = {
    0: (0,     0,    "No structural damage"),
    1: (1,     100,  "Micro thermal damage (< 100 um)"),
    2: (100,   600,  "Melting and crumbling (100 to 600 um)"),
    3: (600,   950,  "Vaporisation crater (600 to 950 um)"),
    4: (950, 99999,  "Catastrophic damage (> 950 um)"),
}

# ── Colores BGR para overlay ────────────────────────────────────────────────
C_CRACK        = (0,   0, 255)   # Rojo   → Grietas
C_CRATER       = (0, 220,  30)   # green for craters
C_EDGE_ZONE    = (100, 100, 0)   # Oliva  → Zona canto
C_ARTIFACT     = (50,  50, 50)   # Gris   → Artefacto filtrado


# ── GLOBAL CONFIGURATION (HYPERPARAMETERS) ──────────────────────────────────
# These values govern the behaviour of the whole segmentation. They are the set found
# by the evolutionary search, and can be overwritten from outside, which is how
CFG = {
    # I. Motor Watershed (optimized by genetic algorithm)
    "ws_open_init": 5, "ws_min_sum": 77, "ws_close": 14, 
    "ws_open_sec": 2, "ws_seed_pct": 30, "ws_firewall": 0.02894560873600488,
    
    # II. Crater detectors (optimized by genetic algorithm)
    "dark_sens": -0.43860267241600265, "dark_open": 20, "dark_exp": 85,
    "bright_sens": 1.175115477101771, "bright_close": 2, "bright_open": 10, "bright_exp": 70,
    "abl_blur": 25, "abl_sens": 0.457654330975265, "abl_open": 7, "abl_exp": 391,
    "rough_grad": 3, "rough_std": 1.6390138273973953, "rough_open": 7, "rough_exp": 62,
    
    # III. Post-Procesado y Artefactos (optimized by genetic algorithm)
    "min_crater_area": 487, "vapor_ring": 7, "vapor_circ": 0.3606198609338688,
    "art_zone1": 0.3674023140184599, "art_zone2": 0.326550447357903, "art_int": 100,
    
    # IV. Grietas (optimized by genetic algorithm)
    "frangi_start": 0.5908931246739, "frangi_stop": 9.187282245657343, "bh_kernel": 19,
    "crack_h_out": 0.08051255424535078, "crack_l_out": 0.007429075583897759,
    "crack_h_in": 0.25754780990041803, "crack_l_in": 0.06410511242262029,
    "bridge_h": 5, "bridge_v": 15, "min_crack_area": 23,
    "surv_v": 25, "surv_h": 65, "min_skel": 14,
 
}

# ─────────────────────────────────────────────────────────────────────────────
#  1. CARGA Y ESCALA
# ─────────────────────────────────────────────────────────────────────────────

def load_image(path):
    """Carga imagen SEM y retorna (bgr, gray)."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"No se puede leer: {path}")
    return img, cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def detect_meta_strip(gray):
    """
    Locate the SEM metadata bar at the bottom of the frame,
    (fondo oscuro con texto blanco).
    Returns its height in pixels.
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


def auto_scale(gray, meta_h):
    """
    Read the magnification, or the physical width, from the SEM metadata bar
    using OCR with some image preprocessing.
    """
    try:
        import pytesseract
        from PIL import Image
        import re
        import cv2

        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        strip = gray[-meta_h:, :]
        
        _, thresh = cv2.threshold(strip, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresh = cv2.bitwise_not(thresh)
        
        h_s, w_s = thresh.shape
        thresh = cv2.resize(thresh, (w_s * 3, h_s * 3), interpolation=cv2.INTER_CUBIC)

        pil = Image.fromarray(thresh)
        text = pytesseract.image_to_string(pil, config='--psm 6').replace('\n', ' ')
        
        w_px = gray.shape[1] 

        # STRATEGY 1: look for the magnification ("Mag = 500 X")
        m_mag = re.search(r'[Mm]ag\s*[=:]?\s*(\d+)', text)
        if m_mag:
            mag_val = float(m_mag.group(1))
            
            if mag_val == 500:
                sc = 0.2236
            elif mag_val == 75:
                sc = 1.4941
            elif mag_val > 0:
                sc = 111.8 / mag_val
            else:
                sc = None
                
            if sc is not None:
                print(f"  [auto-scale OCR] magnification {mag_val:.0f}x -> {sc:.4f} um/px")
                return sc

        # STRATEGY 2: look for the physical width ("Width = 229 um")
        for pat, mult in [
            (r'[Ww]idth\s*[=:]\s*([\d.]+)\s*mm',    1000.0),
            (r'[Ww]idth\s*[=:]\s*([\d.]+)\s*[µu]m',    1.0),
        ]:
            m = re.search(pat, text)
            if m:
                um = float(m.group(1)) * mult
                sc = um / w_px
                print(f"  [auto-scale OCR] physical width {um:.1f} um -> {sc:.4f} um/px")
                return sc

    except Exception as e:
        print(f"  [OCR error] Tesseract failed: {e}")
        pass
        
    return None


# ─────────────────────────────────────────────────────────────────────────────
#  2. ANALYSIS ZONES (the "anchor and step back" scanner)
# ─────────────────────────────────────────────────────────────────────────────

def compute_zones_dynamic(roi_gray, bottom_pct=0.05):
    """
    Locate the material edge from TEXTURE and structural gradient.
    Whether the background is white, black or diffuse is irrelevant here. What the
    scanner looks for is the row where the natural roughness of tungsten begins.
    """
    import numpy as np
    from scipy import ndimage
    import cv2
    
    h, w = roi_gray.shape
    artifact_start = int(h * (1.0 - bottom_pct))
    
    max_search = int(h * 0.35)
    search_roi = roi_gray[:max_search, :]
    
    # 1. Filtro de Textura (Gradiente de Sobel masivo)
    blur = cv2.GaussianBlur(search_roi, (9, 9), 0)
    sobel_y = np.abs(cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=5))
    sobel_x = np.abs(cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=5))
    texture_map = sobel_x + sobel_y
    
    # 2. Perfil horizontal de textura (media por fila)
    texture_profile = np.mean(texture_map, axis=1)
    texture_smooth = ndimage.gaussian_filter1d(texture_profile, sigma=3.0)
    
    safe_top = 10
    
    if len(texture_smooth) > safe_top:
        grad_texture = np.diff(texture_smooth)
        
        # --- MEJORA SOBRE TU BASE: FILTRO ANTI-DOBLE FONDO ---
        max_grad = np.max(grad_texture[safe_top:])
        max_t = np.max(texture_smooth)
        min_t = np.min(texture_smooth)
        
        # Treat as "real material" anything above 40% of the texture range
        umbral_material = min_t + (max_t - min_t) * 0.40
        
        # Collect every gradient peak worth considering
        picos_candidatos = np.where(grad_texture[safe_top:] > max_grad * 0.35)[0] + safe_top
        
        peak_edge = None
        # Walk the peaks from the bottom up, to take the one closest to the material
        for pico in reversed(picos_candidatos):
            # Survival condition: ten pixels below the jump the texture must still
            # be high, which is what distinguishes tungsten from a stray line in the
            # background.
                peak_edge = pico
                break
                
        # If nothing survives the filter, fall back to the plain argmax
        if peak_edge is None:
            peak_edge = int(np.argmax(grad_texture[safe_top:])) + safe_top
        # -----------------------------------------------------
        
        if (max_t - min_t) > 10.0 and texture_smooth[peak_edge] > min_t + 5.0:
            # The slope has been found. Add a small margin of three pixels so that
            # the barrier settles cleanly below it.
            edge_end = peak_edge + 3
        else:
            edge_end = 0
    else:
        edge_end = 0
        
    return int(min(max_search, edge_end)), int(artifact_start)

# ─────────────────────────────────────────────────────────────────────────────
#  3. CRACK DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def detect_cracks(roi_gray, crater_mask, edge_end, artifact_start,
                  frangi_thresh=0.18, min_skel_px=15, min_area_px=45):
    
    """
    Extract the crack network from a SEM micrograph through ridge energy maps,
    a Frangi filter combined with a black-hat transform, and topological filtering.

    Binarisation is by dual hysteresis. Sensitivity stays high on sound material,
    so that micro-cracks survive, while stricter thresholds apply inside craters,
    where the internal roughness of an ablated region would otherwise be picked up
    as a tangle of spurious cracks.

    Parameters
    -----------
    roi_gray : numpy.ndarray
        Greyscale region of interest, with the metadata bar already removed.
    crater_mask : numpy.ndarray o None
        Boolean mask delimiting the craters. It switches on the internal
        topological lock that separates real cracks from ablation noise.
    edge_end : int
        Row index marking the lower limit of the edge zone.
    artifact_start : int
        Row index at which the bottom artefact band starts.
    frangi_thresh : float, opcional (por defecto = 0.18)
        Threshold for the hysteresis applied inside craters. Higher values reject
        more false positives but may lose very faint cracks.
    min_skel_px : int, opcional (por defecto = 15)
        Minimum length in pixels that the resulting skeleton must reach
        para sobrevivir al filtro final.
    min_area_px : int, opcional (por defecto = 45)
        Minimum area in pixels that a crack blob must have before it is
        skeletonised.

    Retorna:
    --------
    tuple de numpy.ndarray:
        - mask_out:   boolean mask of the dilated skeleton, for the visual overlay
        - skel_out:   boolean mask of the pure one-pixel skeleton, for measurement
        - labels_out: int32 array giving each independent crack a unique id
    """
    from skimage.filters import frangi, apply_hysteresis_threshold
    from skimage.morphology import skeletonize, binary_dilation, disk, remove_small_objects
    from skimage.measure import regionprops, label as ski_label
    import cv2
    import numpy as np
    import random
    from pathlib import Path

    h, w = roi_gray.shape

    # 1. CLAHE + Micro-Suavizado
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_cl = clahe.apply(roi_gray)
    img_cl = cv2.medianBlur(img_cl, 3)

    # 2. FRANGI AMPLIADO
    img_inv = (255 - img_cl).astype(np.float32) / 255.0
    tubes = frangi(img_inv, sigmas=np.arange(CFG["frangi_start"], CFG["frangi_stop"], 0.5), alpha=0.5, beta=0.5, black_ridges=False)
    if tubes.max() > 0:
        tubes = tubes / tubes.max()

    # 3. SYMMETRIC BLACK-HAT
    blur_gray = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    kernel_bh = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CFG["bh_kernel"], CFG["bh_kernel"]))
    blackhat = cv2.morphologyEx(blur_gray, cv2.MORPH_BLACKHAT, kernel_bh)
    
    bh_norm = blackhat.astype(np.float32)
    if bh_norm.max() > 0:
        bh_norm = bh_norm / bh_norm.max()

    # 4. ENERGY MAP
    crack_energy = tubes * bh_norm
    if crack_energy.max() > 0:
        crack_energy = crack_energy / crack_energy.max()

    
    # 5. HYSTERESIS DUAL 
    high_out = CFG["crack_h_out"]
    low_out  = CFG["crack_l_out"]
    mask_out = apply_hysteresis_threshold(crack_energy, low_out, high_out)
    
    mask_hyst = mask_out.copy()
    
    has_crater = crater_mask is not None and np.any(crater_mask)
    if has_crater:
        high_in = max(CFG["crack_h_in"], frangi_thresh + 0.05)
        low_in  = CFG["crack_l_in"]
        mask_in = apply_hysteresis_threshold(crack_energy, low_in, high_in)
        
        crater_zone = binary_dilation(crater_mask.astype(bool), disk(3))
        mask_hyst[crater_zone] = mask_in[crater_zone]

    # 6. PUENTE DE COSTURA ORIGINAL 
    kernel_bridge = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CFG["bridge_h"], CFG["bridge_v"]))
    mask_closed = cv2.morphologyEx(mask_hyst.astype(np.uint8), cv2.MORPH_CLOSE, kernel_bridge)

    if edge_end > 0:
        mask_closed[:edge_end, :] = False

    mask_cln = remove_small_objects(mask_closed.astype(bool), min_size=CFG["min_crack_area"])

   
    # 7. TOPOLOGICAL ROOT FILTER
   
    labels_raw = ski_label(mask_cln)
    props = regionprops(labels_raw)
    
    mask_final = np.zeros_like(mask_cln, dtype=bool)
    
    for pp in props:
        minr, minc, maxr, maxc = pp.bbox
        height_bb = maxr - minr
        width_bb = maxc - minc + 1
        area = pp.area
        r, c = pp.coords[:, 0], pp.coords[:, 1]
        
        
        if has_crater:
            in_crater = np.sum(crater_zone[r, c])
            if in_crater > 0:
                out_crater = area - in_crater
                if out_crater < 30:
                    continue
        
       
        if width_bb > height_bb * 2.5 and height_bb < 15:
            continue
            
        dist_to_surface = minr - edge_end
        
        if dist_to_surface > CFG["surv_v"]:
            longitud = max(height_bb, width_bb)
            grosor = area / longitud if longitud > 0 else 0
            
            if grosor > 5.5 or pp.eccentricity < 0.80:
                continue
            
            is_vertical_tail = height_bb > (width_bb * 1.2)
            min_survival_height = CFG["surv_v"] if is_vertical_tail else CFG["surv_h"]
            
            if height_bb < min_survival_height:
                continue
            
        mask_final[r, c] = True

    # 8. FINAL SKELETONISATION
    skel_raw = skeletonize(mask_final)
    labels_skel = ski_label(skel_raw)
    props_skel = regionprops(labels_skel)

    skel_out = np.zeros_like(skel_raw, dtype=bool)
    labels_out = np.zeros_like(labels_skel, dtype=np.int32)
    new_lbl = 1

    for pp in props_skel:
        if pp.area < min_skel_px:
            continue
            
        r, c = pp.coords[:, 0], pp.coords[:, 1]
        skel_out[r, c] = True
        labels_out[r, c] = new_lbl
        new_lbl += 1

    mask_out = binary_dilation(skel_out, disk(1))

    # MODO DEBUG
    #ruta_script = Path(__file__).parent.absolute()
    #debug_dir = ruta_script / "DEBUG_CRACKS"
    #debug_dir.mkdir(parents=True, exist_ok=True)
    #rid = random.randint(1000, 9999)

    
    #cv2.imwrite(str(debug_dir / f"{rid}_1_Mapa_Energia.png"), (crack_energy * 255).astype(np.uint8))
    #cv2.imwrite(str(debug_dir / f"{rid}_2_Mascara_PreFiltro.png"), mask_cln.astype(np.uint8) * 255)
    #cv2.imwrite(str(debug_dir / f"{rid}_3_Esqueleto_Final.png"), skel_out.astype(np.uint8) * 255)

    return mask_out, skel_out, labels_out


def classify_cracks(labels, skel, edge_end, um_per_px):
    """
    Classify the detected cracks and derive the metrics used in the paper.

    LARGE crack (reaches the edge zone):
      0 = no hay
      1 = longitud < 1000 µm
      2 = longitud ≥ 1000 µm

    SMALL cracks (internal, not connected to the edge):
      0 = no hay
      1 = 1–2 grietas
      2 = > 2 grietas

    REGULAR network (uniform spacing between edge cracks):
      0 = no
      1 = yes (at least three edge cracks whose spacing has a coefficient of

    Retorna dict completo.
    """
    large_list, small_list = [], []

    for pp in regionprops(labels):
        r, c    = np.where(labels == pp.label)
        sk_px   = int(skel[r, c].sum())
        length  = sk_px * um_per_px
        in_edge = bool(np.any(r < edge_end))

        entry = {
            'label':         pp.label,
            'skel_px':       sk_px,
            'length_um':     round(length, 1),
            'centroid':      pp.centroid,
            'bbox':          pp.bbox,
            'connects_edge': in_edge,
        }
        (large_list if in_edge else small_list).append(entry)

    # Grieta grande 
    large_max_um = max((x['length_um'] for x in large_list), default=0.0)
    if large_max_um < 1:
        large_cls = 0
    elif large_max_um < 1000:
        large_cls = 1
    else:
        large_cls = 2

    # Small cracks
    n_small   = len(small_list)
    small_cls = 0 if n_small == 0 else (1 if n_small <= 2 else 2)

    # Regularity, from the X positions at the surface with a permissive CV threshold.
    # Small cracks close to the surface count too, not only the "large" ones.
    regular_cls = 0
    regular_cv = None
    surface_margin = max(40, int(labels.shape[0] * 0.08))

    near_surface = [
        x for x in (large_list + small_list)
        if x['connects_edge'] or x['bbox'][0] < (edge_end + surface_margin)
    ]

    if len(near_surface) >= 3:
        surface_x = []
        for crack in near_surface:
            lbl = crack['label']
            ys, xs = np.where(labels == lbl)
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
                cv = float(np.std(spacings) / (mean_sp + 1e-9))
                regular_cv = round(cv, 3)
                if cv < 0.45:
                    regular_cls = 1

    total_um = sum(x['length_um'] for x in large_list + small_list)

    return {
        'large_crack_class':   large_cls,
        'large_crack_max_um':  round(large_max_um, 1),
        'small_crack_class':   small_cls,
        'regular_crack_class': regular_cls,
        'regular_cv':          regular_cv,
        'n_large':             len(large_list),
        'n_small':             n_small,
        'total_length_um':     round(total_um, 1),
        'large_list':          large_list,
        'small_list':          small_list,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  4. CRATER DETECTION
# ─────────────────────────────────────────────────────────────────────────────

def local_relief(gray):
    h, w = gray.shape
    k_size = (int(w * 0.45) // 2) * 2 + 1 
    bg = cv2.GaussianBlur(gray, (k_size, k_size), 0)
    
    relief = gray.astype(np.float32) - bg.astype(np.float32)
    std_val = np.std(relief) + 1e-6
    relief_norm = (relief - np.mean(relief)) / std_val
    return relief_norm

def watershed_from_mask(relief_norm, mask, roi_gray, edge_end, expand_limit=100):
    from skimage.morphology import disk
    from scipy.ndimage import binary_fill_holes
    import cv2
    import numpy as np

    h, w = roi_gray.shape
    if edge_end > 0 and edge_end < (h * 0.30):
        mask[:edge_end, :] = 0

    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, disk(CFG["ws_open_init"]))

    if np.sum(mask) < CFG["ws_min_sum"]:
        return np.zeros_like(mask, bool)

    mask_merged = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, disk(CFG["ws_close"]))
    mask_merged = binary_fill_holes(mask_merged).astype(np.uint8)
    mask_merged = cv2.morphologyEx(mask_merged, cv2.MORPH_OPEN, disk(CFG["ws_open_sec"]))

    dist = cv2.distanceTransform(mask_merged, cv2.DIST_L2, 5)
    valid = dist[dist > 0]
    if len(valid) == 0:
        return np.zeros_like(mask, bool)

    thresh = np.percentile(valid, CFG["ws_seed_pct"])
    _, fg = cv2.threshold(dist, thresh, 255, 0)
    
    _, markers = cv2.connectedComponents(fg.astype(np.uint8))
    markers = markers + 1
    
    # 1. Apply the dynamic area limit (200 for ablation, 45 to 60 otherwise)
    dilated_mask = cv2.dilate(mask_merged, disk(expand_limit), iterations=1)
    
    # 2. Topographic firewall, which stops the flooding from bleeding downwards.
    if relief_norm is not None:
        # Anything with a relief below ws_firewall counts as sound valley or plain,
        # and the growing region is denied permission to travel across it.
        # The flood hits this invisible barrier and stops.
        dilated_mask[relief_norm < CFG["ws_firewall"]] = 0

    unknown = cv2.subtract(dilated_mask, fg.astype(np.uint8))
    markers[unknown > 0] = 0

    blur = cv2.GaussianBlur(roi_gray, (15, 15), 0)
    grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    grad_mag = cv2.GaussianBlur(grad_mag, (5, 5), 0) 
    grad_norm = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    
    grad_rgb = cv2.cvtColor(grad_norm, cv2.COLOR_GRAY2BGR)
    ws = cv2.watershed(grad_rgb, markers)

    return ws > 1

def watershed_dark(roi_gray, relief_norm, edge_end):
    from skimage.morphology import disk
    import cv2
    mask = relief_norm < CFG["dark_sens"]
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, disk(CFG["dark_open"]))
    # Desmoronamiento oscuro: muy conservador para no invadir
    return watershed_from_mask(relief_norm, mask, roi_gray, edge_end, expand_limit=CFG["dark_exp"])

def watershed_bright(roi_gray, relief_norm, edge_end):
    from skimage.morphology import disk
    from scipy.ndimage import binary_fill_holes
    import cv2
    mask = relief_norm > CFG["bright_sens"]
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, disk(CFG["bright_close"]))
    mask = binary_fill_holes(mask).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, disk(CFG["bright_open"]))
    # Desmoronamiento blanco: conservador
    return watershed_from_mask(relief_norm, mask, roi_gray, edge_end, expand_limit=CFG["bright_exp"])

def detect_ablation(roi_gray, relief_norm, edge_end):
    from skimage.morphology import disk
    import cv2
    smooth = cv2.GaussianBlur(relief_norm, (CFG["abl_blur"], CFG["abl_blur"]), 0)
    mask = np.abs(smooth) > CFG["abl_sens"]
    h, w = roi_gray.shape
    if edge_end > 0 and edge_end < (h * 0.30):
        mask[:edge_end, :] = 0 
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, disk(CFG["abl_open"]))
    # Large ablation: free to expand, held back only by the firewall
    return watershed_from_mask(relief_norm, mask, roi_gray, edge_end, expand_limit=CFG["abl_exp"])

def watershed_rough(roi_gray, edge_end):
    from skimage.morphology import disk
    import cv2
    import numpy as np
    blur = cv2.GaussianBlur(roi_gray, (5, 5), 0)
    grad = cv2.morphologyEx(blur, cv2.MORPH_GRADIENT, disk(CFG["rough_grad"]))
    grad_smooth = cv2.GaussianBlur(grad, (21, 21), 0)
    mask = grad_smooth > (np.mean(grad_smooth) + np.std(grad_smooth) * CFG["rough_std"])
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, disk(CFG["rough_open"]))
    # Standard roughness channel
    return watershed_from_mask(None, mask, roi_gray, edge_end, expand_limit=CFG["rough_exp"])

def build_crater_objects(roi_gray, labels):

    from skimage.measure import regionprops
    from skimage.morphology import binary_dilation, disk
    import numpy as np

    craters = []
    for pp in regionprops(labels, intensity_image=roi_gray):
        area_px = int(pp.area)
        
        if area_px < CFG["min_crater_area"]:
            continue

        cy, cx = pp.centroid
        mask = labels == pp.label

        dilated_mask = binary_dilation(mask, disk(CFG["vapor_ring"]))
        ring_mask = dilated_mask & ~mask
        mean_inside = np.mean(roi_gray[mask])
        mean_ring = np.mean(roi_gray[ring_mask]) if np.any(ring_mask) else mean_inside

        perimeter = pp.perimeter + 1e-6
        circularity = (4 * np.pi * area_px) / (perimeter**2)

        if mean_inside > mean_ring and circularity > CFG["vapor_circ"]:
            ctype = 'vaporization'
        else:
            ctype = 'collapse'

        diam = getattr(pp, 'equivalent_diameter_area', getattr(pp, 'equivalent_diameter', 0.0))

        craters.append({
            'label': pp.label,
            'area_px': area_px,
            'equiv_diam_px': float(diam),
            'centroid': (float(cy), float(cx)),
            'bbox': tuple(int(v) for v in pp.bbox),
            'eccentricity': float(pp.eccentricity),
            'solidity': float(pp.solidity),
            'crater_type': ctype,
            'mean_intensity': float(mean_inside),
            'mask': mask
        })
    return craters

def detect_craters(roi_gray, crack_mask, edge_end, artifact_start):

    from skimage.measure import label as ski_label
    from skimage.morphology import remove_small_objects
    import cv2
    import numpy as np
    import random
    from pathlib import Path
    
    relief_norm = local_relief(roi_gray)

    dark_ws   = watershed_dark(roi_gray, relief_norm, edge_end)
    bright_ws = watershed_bright(roi_gray, relief_norm, edge_end)
    ablation  = detect_ablation(roi_gray, relief_norm, edge_end)
    rough_ws  = watershed_rough(roi_gray, edge_end) 

    crater_mask = dark_ws | bright_ws | ablation | rough_ws

    h, w = roi_gray.shape
    if edge_end > 0:
        crater_mask[:edge_end, :] = False
        
    if artifact_start > 0:
        crater_mask[artifact_start:, :] = False
    
    if crack_mask is not None:
        crater_mask[crack_mask.astype(bool)] = False

    crater_mask = remove_small_objects(crater_mask, min_size=CFG["min_crater_area"])

    labels_finales = ski_label(crater_mask)
    raw_craters = build_crater_objects(roi_gray, labels_finales)

    # MODO DEBUG 
    #ruta_script = Path(__file__).parent.absolute()
    #debug_dir = ruta_script / "DEBUG_MULTI_WS"
    #debug_dir.mkdir(parents=True, exist_ok=True)
    #rid = random.randint(1000, 9999)

    
    #relief_vis = cv2.normalize(relief_norm, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    #cv2.imwrite(str(debug_dir / f"{rid}_1_Relieve.png"), relief_vis)
    #cv2.imwrite(str(debug_dir / f"{rid}_2_Colapso_Oscuro.png"), dark_ws.astype(np.uint8)*255)
    #cv2.imwrite(str(debug_dir / f"{rid}_3_Contraste_Claro.png"), bright_ws.astype(np.uint8)*255)
    #cv2.imwrite(str(debug_dir / f"{rid}_4_Ablacion.png"), ablation.astype(np.uint8)*255)
    #cv2.imwrite(str(debug_dir / f"{rid}_5_Mascara_Fusionada_Final.png"), crater_mask.astype(np.uint8)*255)

    return raw_craters, labels_finales
    

def filter_artifacts(craters, artifact_start):
    valid = []
    artifacts = [] 
    
    if not craters:
        return valid, artifacts

    h_aprox = artifact_start

    for c in craters:
        area = c['area_px']
        min_r = c['bbox'][0]           
        mean_int = c['mean_intensity'] 
        
        is_dust_or_fragment = False

        
        if area < 600:
            is_dust_or_fragment = True
        else:
            if min_r > (h_aprox * CFG["art_zone1"]):
                is_dust_or_fragment = True
            elif min_r > (h_aprox * CFG["art_zone2"]) and mean_int < CFG["art_int"]:
                is_dust_or_fragment = True

        if not is_dust_or_fragment:
            valid.append(c)

    
    return valid, []


def classify_craters(craters, um_per_px):
    """
    Assign a grade from 0 to 4 to each crater from its equivalent diameter in um.
    Retorna (max_grade, details_list).
    """
    if not craters:
        return 0, []

    details = []
    for c in craters:
        d_um  = c['equiv_diam_px'] * um_per_px
        a_um2 = c['area_px'] * (um_per_px ** 2)
        if d_um < 1:      g = 0
        elif d_um < 100:  g = 1
        elif d_um < 600:  g = 2
        elif d_um < 950:  g = 3
        else:             g = 4
        details.append({
            **c,
            'diam_um':    round(d_um, 1),
            'area_um2':   round(a_um2, 1),
            'grade':      g,
            'grade_desc': CRATER_GRADES[g][2],
        })
    return max(d['grade'] for d in details), details


def build_overlay(img_bgr, h_roi, crack_mask, crack_skel, crack_labels,
                  crack_info, crater_details, dark_labels,
                  edge_end, artifact_start, artifacts):
    """
    Build the colour-coded overlay:
      Rojo   → Todas las grietas
      green  -> all craters
      Oliva  → Zona de canto excluida (semitransparente)
      Gris   → Artefacto filtrado (solo bounding box)
    """
    overlay = img_bgr.copy()
    alpha   = 0.3

    # Zona de canto (
    if edge_end > 0:
        tmp = overlay[:edge_end, :].copy()
        overlay[:edge_end, :] = cv2.addWeighted(
            tmp, 0.72,
            np.full_like(tmp, C_EDGE_ZONE), 0.28, 0)

    # Craters
    for cd in crater_details:
        col  = C_CRATER
        mask = (dark_labels == cd['label']).astype(np.uint8)
        col_layer = np.zeros_like(img_bgr)
        col_layer[mask > 0] = col
        overlay[:h_roi] = cv2.addWeighted(
            overlay[:h_roi], 1 - alpha, col_layer[:h_roi], alpha, 0)
        cntrs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, cntrs, -1, col, 2)
        cy, cx = cd['centroid']
        cv2.putText(overlay, f"G{cd['grade']}",
                    (int(cx) + 5, int(cy) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

    # Grietas 
    for r, c in np.argwhere(crack_mask[:h_roi]):
        overlay[r, c] = C_CRACK

    # Esqueleto 
    for r, c in np.argwhere(crack_skel[:h_roi]):
        cv2.circle(overlay, (c, r), 1, C_CRACK, -1)

    # Artefactos 
    for a in artifacts:
        bb = a['bbox']
        cv2.rectangle(overlay, (bb[1], bb[0]), (bb[3], bb[2]), C_ARTIFACT, 1)

    return overlay


def save_figure(img_bgr, overlay, crack_info, max_grade, crater_details,
                um_per_px, edge_end, artifact_start, out_dir, fname):
    """Comparison figure with the classification summary alongside."""
    bg = '#141428'
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    fig.patch.set_facecolor(bg)
    for ax in axes:
        ax.set_facecolor(bg)
        for sp in ax.spines.values():
            sp.set_color('white')
        ax.tick_params(colors='white')

    axes[0].imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB), cmap='gray')
    axes[0].set_title('Imagen original SEM', color='white', fontsize=13)
    if edge_end > 0:
        axes[0].axhline(edge_end, color='yellow', lw=1, ls='--', alpha=0.7,
                        label='Edge zone limit')
    axes[0].axhline(artifact_start, color='gray', lw=1, ls='--', alpha=0.7,
                    label='Zona artefactos')
    axes[0].legend(loc='lower right', fontsize=8, facecolor=bg,
                   labelcolor='white', framealpha=0.7)

    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Damage segmentation', color='white', fontsize=13)

    # LEYENDA SIMPLIFICADA
    leg_patches = [
        mpatches.Patch(color=np.array(C_CRACK[::-1]) / 255, label='Grietas'),
        mpatches.Patch(color=np.array(C_CRATER[::-1]) / 255, label='Craters'),
        mpatches.Patch(color=(0.39, 0.39, 0.0), label='Zona canto (excluida)'),
    ]
    axes[1].legend(handles=leg_patches, loc='lower right', fontsize=8,
                   facecolor=bg, labelcolor='white', framealpha=0.7)

    ci   = crack_info
    gd   = CRATER_GRADES.get(max_grade, (0, 0, ''))[2]
    cv_str = f"  [CV={ci['regular_cv']:.3f}]" if ci.get('regular_cv') is not None else ""
    txt  = (
        "----------  SEVERITY CLASSIFICATION  ----------\n"
        f"Escala: {um_per_px:.4f} µm/px\n\n"
        "GRIETAS\n"
        f"  Grieta grande    clase {ci['large_crack_class']}"
        f"  (max {ci['large_crack_max_um']:.0f} µm, n={ci['n_large']})\n"
        f"  Small cracks, class {ci['small_crack_class']}"
        f"  (n={ci['n_small']})\n"
        f"  Grietas regulares clase {ci['regular_crack_class']}"
        f"  ({'YES' if ci['regular_crack_class'] else 'NO'}){cv_str}\n"
        f"  Longitud total: {ci['total_length_um']:.0f} µm\n\n"
        "CRATERS\n"
        f"  Maximum grade: {max_grade}  =  {gd}\n"
        f"  Craters detected: {len(crater_details)}"
    )
    fig.text(0.5, 0.01, txt, ha='center', va='bottom',
             fontsize=9, color='white', fontfamily='monospace',
             bbox=dict(boxstyle='round,pad=0.7', facecolor='#0a0a20',
                       edgecolor='#4488ff', alpha=0.93))

    plt.tight_layout(rect=[0, 0.23, 1, 1])
    out = str(out_dir / f"{Path(fname).stem}_analisis.png")
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor=bg)
    plt.close(fig)
    print(f"  → Figura:  {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  6. INFORME JSON
# ─────────────────────────────────────────────────────────────────────────────

def save_json_report(crack_info, max_grade, crater_details,
                     um_per_px, out_dir, fname):
    report = {
        'file': fname,
        'um_per_px': um_per_px,
        'cracks': {
            'large_crack_class':   crack_info['large_crack_class'],
            'large_crack_max_um':  crack_info['large_crack_max_um'],
            'small_crack_class':   crack_info['small_crack_class'],
            'regular_crack_class': crack_info['regular_crack_class'],
            'regular_cv':          crack_info.get('regular_cv'),
            'n_large':             crack_info['n_large'],
            'n_small':             crack_info['n_small'],
            'total_length_um':     crack_info['total_length_um'],
        },
        'craters': {
            'max_grade':      max_grade,
            'max_grade_desc': CRATER_GRADES.get(max_grade, (0, 0, ''))[2],
            'n_craters':      len(crater_details),
            'list': [
                {'id': d['label'], 'type': d['crater_type'],
                 'grade': d['grade'], 'diam_um': d['diam_um'],
                 'area_um2': d['area_um2']}
                for d in crater_details
            ],
        },
    }
    out = str(out_dir / f"{Path(fname).stem}_informe.json")
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    #print(f"  → JSON:    {out}")
    return out, report


# ─────────────────────────────────────────────────────────────────────────────
#  7. PIPELINE PRINCIPAL
# ─────────────────────────────────────────────────────────────────────────────

def analyze(img_path, um_per_px=None,
            bottom_pct=0.02,
            frangi_thresh=0.18, min_crack_um=15.0,
            output_dir=None, verbose=True,):
    """
    Analyse one OLMAT tungsten SEM micrograph.

    Returns a dict with the classification results and the paths written.
    """
    img_path = Path(img_path)
    out_dir  = Path(output_dir) if output_dir else img_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"\n{'='*62}\n  {img_path.name}\n{'='*62}")

    # ── Carga ────────────────────────────────────────────────────────
    img_bgr, img_gray = load_image(img_path)
    H, W = img_gray.shape
    if verbose:
        print(f"  Imagen: {W}×{H} px")

    # ── Metadatos ────────────────────────────────────────────────────
    meta_h = detect_meta_strip(img_gray)
    if verbose:
        print(f"  Barra metadatos: {meta_h} px")

    # ── Escala ───────────────────────────────────────────────────────
    if um_per_px is None:
        um_per_px = auto_scale(img_gray, meta_h)
    if um_per_px is None:
        # Valor por defecto para 500X OLMAT (1024px × 0.224 µm/px ≈ 229 µm)
        um_per_px = 0.224
        print(f"  [scale] Auto-detection failed. Using {um_per_px} um/px (500X).")
        print("          Pass --scale for a more reliable value.")
    if verbose:
        print(f"  Escala: {um_per_px:.4f} µm/px  "
              f"({W * um_per_px:.0f} × {H * um_per_px:.0f} µm)")

    # ── ROI (sin barra de metadatos) ─────────────────────────────────
    roi_gray = img_gray[:H - meta_h, :]
    roi_bgr  = img_bgr [:H - meta_h, :]
    h_roi    = roi_gray.shape[0]

    edge_end, artifact_start = compute_zones_dynamic(roi_gray, bottom_pct=0.02)
    if verbose:
        print(f"  Zona canto: 0–{edge_end} px "
              f"({edge_end * um_per_px:.0f} µm)")
        print(f"  ROI interior: {edge_end}–{artifact_start} px")

    # ── Craters first ────────────────────────────────────────────────
    if verbose:
        print("\n  [craters]")

    # First pass with an empty crack mask
    empty_crack_mask = np.zeros_like(roi_gray, dtype=bool)

    raw_craters, dark_labels = detect_craters(
        roi_gray, empty_crack_mask, edge_end, artifact_start)

    valid_craters, artifacts = filter_artifacts(raw_craters, artifact_start)
    max_grade, crater_details = classify_craters(valid_craters, um_per_px)

    # Crear crater_mask a partir de dark_labels
    ablation_labels = [c['label'] for c in crater_details if c['crater_type'] == 'vaporization']
    ablation_mask = np.isin(dark_labels, ablation_labels)

    if verbose:
        gd = CRATER_GRADES.get(max_grade, (0, 0, ''))[2]
        print(f"    Valid: {len(crater_details)}  "
            f"Artefactos filtrados: {len(artifacts)}")
        print(f"    Maximum grade: {max_grade} = {gd}")

    # ── Cracks, once the craters are known ───────────────────────────
    if verbose:
        print("\n  [Grietas]")

    min_skel_px = max(10, int(min_crack_um / um_per_px))
    valid_labels = [c['label'] for c in crater_details]
    all_craters_mask = np.isin(dark_labels, valid_labels)

    crack_mask, crack_skel, crack_labels = detect_cracks(
        roi_gray,
        all_craters_mask,            
        edge_end,
        artifact_start,
        frangi_thresh=frangi_thresh,
        min_skel_px=min_skel_px,
        min_area_px=max(20, min_skel_px // 2))

    crack_info = classify_cracks(crack_labels, crack_skel, edge_end, um_per_px)
    ci = crack_info

    if verbose:
        print(f"    Grandes (canto):  n={ci['n_large']}  "
            f"clase={ci['large_crack_class']}  "
            f"max={ci['large_crack_max_um']:.0f} µm")
        print(f"    Small internal: n={ci['n_small']}  "
            f"clase={ci['small_crack_class']}")
        print(f"    Regulares:         clase={ci['regular_crack_class']}")
        print(f"    Longitud total:    {ci['total_length_um']:.0f} µm")

    

    # ── Construir arrays full-image ──────────────────────────────────
    def expand(arr, dtype):
        out = np.zeros((H, W), dtype=dtype)
        out[:h_roi, :] = arr
        return out

    full_mask   = expand(crack_mask,   bool)
    full_skel   = expand(crack_skel,   bool)
    full_labels = expand(crack_labels, int)
    full_dark   = expand(dark_labels,  int)

    # ── Overlay ──────────────────────────────────────────────────────
    overlay = build_overlay(
        img_bgr, h_roi,
        full_mask, full_skel, full_labels,
        crack_info, crater_details, full_dark,
        edge_end, artifact_start, artifacts)

   # ── Guardar ──────────────────────────────────────────────────────
    ov_path = None
    out_fig = True
    rpt = None
    
    if output_dir is not None:
        ov_path = str(out_dir / f"{img_path.stem}_overlay.png")
        cv2.imwrite(ov_path, overlay)
        if verbose: print(f"  → Overlay: {ov_path}")

        out_fig = save_figure(img_bgr, overlay, crack_info, max_grade,
                               crater_details, um_per_px, edge_end, artifact_start,
                               out_dir, img_path.name)
        _, rpt = save_json_report(crack_info, max_grade, crater_details,
                                   um_per_px, out_dir, img_path.name)
    # ── Resumen ──────────────────────────────────────────────────────
    if verbose:
        gd = CRATER_GRADES.get(max_grade, (0, 0, ''))[2]
        print(f"\n{'─'*62}")
        print(f"  RESUMEN  {img_path.name}")
        print(f"{'─'*62}")
        print(f"  Grieta grande    : clase {ci['large_crack_class']} "
              f"({ci['large_crack_max_um']:.0f} µm)")
        print(f"  Small cracks     : class {ci['small_crack_class']} "
              f"(n={ci['n_small']})")
        print(f"  Regulares        : clase {ci['regular_crack_class']}")
        print(f"  Largest crater   : grade {max_grade} = {gd}")
        print(f"{'─'*62}")

    return {
        'image':               str(img_path),
        'um_per_px':           um_per_px,
        'crack_large_class':   ci['large_crack_class'],
        'crack_small_class':   ci['small_crack_class'],
        'crack_regular_class': ci['regular_crack_class'],
        'crater_max_grade':    max_grade,
        'report':              rpt,
        'figure':              out_fig,
        'overlay':             ov_path,
        'mascara_total_booleana': full_mask | (full_dark > 0)
    }


# ─────────────────────────────────────────────────────────────────────────────
#  8. MODO LOTE
# ─────────────────────────────────────────────────────────────────────────────

def batch(input_dir, out_dir=None, **kwargs):
    """Process every SEM image in a directory and write a summary CSV."""
    input_dir = Path(input_dir)
    exts  = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    imgs  = [p for p in sorted(input_dir.iterdir())
             if p.suffix.lower() in exts]
    if not imgs:
        print(f"No images in {input_dir}")
        return []

    if out_dir is None:
        out_dir = input_dir / 'resultados_SEM'
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    kwargs['output_dir'] = str(out_dir)

    results, rows = [], []
    for p in imgs:
        try:
            r = analyze(p, **kwargs)
            results.append(r)
            rows.append({
                'archivo':            p.name,
                'um_per_px':          r['um_per_px'],
                'grieta_grande_cls':  r['crack_large_class'],
                'grieta_peq_cls':     r['crack_small_class'],
                'grieta_reg_cls':     r['crack_regular_class'],
                'crater_grado_max':   r['crater_max_grade'],
                'long_total_um':      r['report']['cracks']['total_length_um'],
                'n_crateres':         r['report']['craters']['n_craters'],
            })
        except Exception as e:
            print(f"  ERROR {p.name}: {e}")
            rows.append({'archivo': p.name, 'error': str(e)})

    csv_path = Path(out_dir) / 'resumen.csv'
    if rows:
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nCSV resumen: {csv_path}")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  9. CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='OLMAT SEM damage analyser for ITER-like tungsten',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    p.add_argument('input',
        help='SEM image (.jpg/.png/.tif) or directory (with --batch)')
    p.add_argument('--scale', type=float, default=None, metavar='UM_PX',
        help='um per pixel (auto-detected when omitted)')
    p.add_argument('--output-dir', default=None,
        help='Output directory')
    p.add_argument('--bottom-pct', type=float, default=0.05,
        help='Bottom fraction treated as the artefact zone (default 0.05)')
    p.add_argument('--crack-thresh', type=float, default=0.18,
        help='Frangi threshold 0 to 1 (default 0.18; higher = fewer false positives)')
    p.add_argument('--min-crack-um', type=float, default=15.0,
        help='Minimum length in um for a crack to be reported (default 15)')
    p.add_argument('--batch', action='store_true',
        help='Process every image in the given directory')
    args = p.parse_args()

    kw = dict(um_per_px=args.scale,
              bottom_pct=args.bottom_pct,
              frangi_thresh=args.crack_thresh,
              min_crack_um=args.min_crack_um,
              output_dir=args.output_dir)

    if args.batch or Path(args.input).is_dir():
        batch(args.input, **kw)
    else:
        analyze(args.input, **kw)



if __name__ == '__main__':
    main()
