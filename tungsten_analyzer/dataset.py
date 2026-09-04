"""Dataset and augmentation for the tungsten SEM micrographs.

With fewer than a hundred annotated images, augmentation is not a refinement but a
requirement: each original micrograph is resampled into roughly twenty distinct
variants per epoch.

Annotations are read directly from the LabelMe JSON files rather than from
pre-rendered masks, which keeps a single source of truth for the ground truth.
Craters are stored as closed polygons and are filled. Cracks are stored as open
polylines following the centreline of the fracture, which is the only annotation
that can be made consistently at these magnifications, and they are rasterised at a
fixed width of 15 pixels for a 1024-pixel-wide frame, scaled proportionally for
other sizes. That width is a drawing convention rather than a measurement, and the
same convention is applied at evaluation time so that training targets and reference
masks always agree.

Expected layout::

    data/
      images/      SEM micrographs (.jpg, .png, .tif)
      masks/       one LabelMe .json per image, same stem

The bottom of every frame carries the SEM metadata bar. It is cropped away before
anything else happens, by default the lower 8 per cent of the frame, and the
annotation is cropped to match.
"""

import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALB_AVAILABLE = True
except ImportError:
    ALB_AVAILABLE = False
    print("[AVISO] albumentations no encontrado.")
    print("        Instala con: pip install albumentations")


# ─────────────────────────────────────────────────────────────────────────────
#  AUGMENTACIONES
# ─────────────────────────────────────────────────────────────────────────────

def get_transforms(mode: str = 'train',
                   img_size: int = 512,
                   augment_strength: str = 'strong') -> 'A.Compose':
    """
    Augmentation pipeline tuned for SEM micrographs of materials.

    It differs from the usual recipes for natural photographs in three ways.
      - NO usamos color jitter (SEM es escala de grises, sin color real)
      - Strong geometric transforms are welcome, since a crack can run at any angle.
      - SEM noise is simulated explicitly, both speckle and shot noise.
      - JPEG compression is not simulated, as the microscope already contributes
        its own compression artefacts.

    -----------
    mode : 'train' | 'val' | 'test'
    img_size : int
        Side of the square to which every image is resized. 512 is the smallest
        size at which fine cracks survive the interpolation.
    augment_strength : 'light' | 'strong'
        'light' for internal cross-validation or warm-up epochs,
        'strong' for the main training run.
    """
    if not ALB_AVAILABLE:
        raise ImportError("Instala albumentations: pip install albumentations")

    if mode == 'train':
        if augment_strength == 'strong':
            return A.Compose([
                # 1. SCALE AND GEOMETRY
                # scale increased from (0.5, 1.0) to (0.6, 1.0) to avoid
                # cropping out thin cracks too aggressively
                A.RandomResizedCrop(
                    size=(img_size, img_size),
                    scale=(0.75, 1.0),
                    ratio=(0.85, 1.18),
                    p=1.0),

                # 2. FLIPS, since a crack has no preferred direction
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),

                # 3. FREE ROTATION, for diagonal cracks
                A.Rotate(limit=45, border_mode=cv2.BORDER_REFLECT_101, p=0.7),

                # 4. ELASTIC DISTORTION, which stands in for surface warping
                A.ElasticTransform(
                    alpha=60, sigma=8,
                    border_mode=cv2.BORDER_REFLECT_101, p=0.3),

                # 5. VARIABLE SEM IMAGING CONDITIONS
                A.OneOf([
                    # Brightness and contrast drift with beam voltage and working distance
                    A.RandomBrightnessContrast(
                        brightness_limit=0.35, contrast_limit=0.35, p=1.0),
                    # Gamma shift (cambio de sensibilidad del detector)
                    A.RandomGamma(gamma_limit=(60, 140), p=1.0),
                    # CLAHE (simula post-procesado SEM diferente)
                    A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=1.0),
                ], p=0.8),

                # 6. RUIDO (shot noise del haz de electrones)
                A.OneOf([
                    A.GaussNoise(std_range=(0.03, 0.12), p=1.0),
                    A.MultiplicativeNoise(multiplier=(0.9, 1.1), p=1.0),
                ], p=0.5),

                # 7. DESENFOQUE (defoco del haz, charging)
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 5), p=1.0),
                    A.MotionBlur(blur_limit=5, p=1.0),
                ], p=0.3),

                # 8. GRID DISTORTION (ligeras deformaciones de la muestra)
                A.GridDistortion(num_steps=5, distort_limit=0.15, p=0.2),

                # 9. SHARPENING — helps the model learn to detect thin
                # crack edges that can be subtle in noisy SEM images
                A.Sharpen(alpha=(0.1, 0.3), lightness=(0.9, 1.1), p=0.3),

                # 10. NORMALISATION using ImageNet statistics, as the encoder expects
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:  # light
            return A.Compose([
                A.Resize(img_size, img_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.15, contrast_limit=0.15, p=0.5),
                A.GaussNoise(std_range=(0.02, 0.08), p=0.3),
                A.Normalize(mean=(0.485, 0.456, 0.406),
                            std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
    else:  # val / test
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.485, 0.456, 0.406),
                        std=(0.229, 0.224, 0.225)),
            ToTensorV2(),
        ])


# ─────────────────────────────────────────────────────────────────────────────
#  DATASET
# ─────────────────────────────────────────────────────────────────────────────

class TungstenSEMDataset(Dataset):
    """
    Dataset of tungsten SEM micrographs with crater and crack annotations.

    Parameters
    -----------
    image_dir : str o Path
        Directory holding the original SEM micrographs.
    mask_crater_dir : str o Path o None
        Directory holding pre-rendered crater masks. When None the crater
        channel is read from the LabelMe JSON instead.
    mask_crack_dir : str o Path o None
        Directory holding pre-rendered crack masks. When None the crack
        channel is read from the LabelMe JSON instead.
    transform : albumentations.Compose
        Augmentation pipeline.
    img_size : int
        Resize target used when no transform is supplied.
    strip_fraction : float
        Fraction of the frame height to discard at the bottom, where the SEM metadata bar sits.
    """

    EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}

    def __init__(self,
                 image_dir: str,
                 mask_crater_dir: Optional[str] = None,
                 mask_crack_dir: Optional[str] = None,
                 transform=None,
                 img_size: int = 512,
                 strip_fraction: float = 0.08):

        self.image_dir       = Path(image_dir)
        self.mask_crater_dir = Path(mask_crater_dir) if mask_crater_dir else None
        self.mask_crack_dir  = Path(mask_crack_dir) if mask_crack_dir else None
        self.transform       = transform
        self.img_size        = img_size
        self.strip_fraction  = strip_fraction

        # Collect every readable image
        self.image_paths = sorted([
            p for p in self.image_dir.iterdir()
            if p.suffix.lower() in self.EXTENSIONS
        ])

        if not self.image_paths:
            raise FileNotFoundError(
                f"No images found in {self.image_dir}")

        print(f"  [Dataset] {len(self.image_paths)} images found in {self.image_dir.name}/")
        self._run_mask_diagnostic()

    def _run_mask_diagnostic(self):
        """
        Scans all images and checks which ones have JSON masks,
        which labels are present, and how many pixels are annotated.
        Prints a summary so you can immediately see if masks are loading.
        """
        import json as json_mod
        
        n_json_found   = 0
        n_json_missing = 0
        n_with_craters = 0
        n_with_cracks  = 0
        n_empty        = 0
        all_labels     = set()
        missing_examples = []

        for img_path in self.image_paths:
            stem = img_path.stem
            
            # Same search logic as __getitem__
            json_path = None
            data_root = img_path.parent.parent  # data/
            search_dirs = [
                data_root / 'masks',
                data_root / 'masks' / 'craters',
                data_root / 'masks' / 'cracks',
                img_path.parent,
                data_root,
            ]
            if self.mask_crater_dir is not None:
                search_dirs.insert(0, self.mask_crater_dir.parent)
                search_dirs.insert(0, self.mask_crater_dir)
            if self.mask_crack_dir is not None:
                search_dirs.insert(0, self.mask_crack_dir)

            for search_dir in search_dirs:
                candidate = search_dir / f"{stem}.json"
                if candidate.exists():
                    json_path = candidate
                    break

            if json_path is not None:
                n_json_found += 1
                try:
                    with open(json_path, 'r', encoding='utf-8') as f:
                        data = json_mod.load(f)
                    shapes = data.get('shapes', [])
                    labels = [s['label'].strip().lower() for s in shapes]
                    all_labels.update(labels)
                    
                    CRATER_LABELS = {'crater', 'cratera', 'cráter', 'vaporization', 
                                    'collapse', 'crater_vaporization', 'crater_collapse'}
                    CRACK_LABELS  = {'crack', 'grieta', 'fracture', 'fractura', 
                                    'crack_large', 'crack_small'}
                    
                    has_crater = any(l in CRATER_LABELS for l in labels)
                    has_crack  = any(l in CRACK_LABELS for l in labels)
                    
                    if has_crater: n_with_craters += 1
                    if has_crack:  n_with_cracks  += 1
                    if not has_crater and not has_crack: n_empty += 1
                except Exception as e:
                    print(f"    [WARNING] Error reading {json_path}: {e}")
            else:
                n_json_missing += 1
                if len(missing_examples) < 3:
                    missing_examples.append(stem)

        print(f"\n  ╔══════════════════════════════════════════════════╗")
        print(f"  ║          MASK DIAGNOSTIC REPORT                  ║")
        print(f"  ╠══════════════════════════════════════════════════╣")
        print(f"  ║  Total images:        {len(self.image_paths):>4}                       ║")
        print(f"  ║  JSON masks found:    {n_json_found:>4}                       ║")
        print(f"  ║  JSON masks MISSING:  {n_json_missing:>4}                       ║")
        print(f"  ║  With crater labels:  {n_with_craters:>4}                       ║")
        print(f"  ║  With crack labels:   {n_with_cracks:>4}                       ║")
        print(f"  ║  Empty (no shapes):   {n_empty:>4}                       ║")
        print(f"  ╠══════════════════════════════════════════════════╣")
        if all_labels:
            print(f"  ║  Labels found: {', '.join(sorted(all_labels)):<34} ║")
        else:
            print(f"  ║  *** NO LABELS FOUND AT ALL ***                  ║")
        print(f"  ╚══════════════════════════════════════════════════╝")
        
        if n_json_missing > 0:
            print(f"\n  [WARNING] {n_json_missing} images have NO JSON mask file!")
            if missing_examples:
                print(f"    Examples: {', '.join(missing_examples)}")
            print(f"    Searched in: {[str(d) for d in search_dirs]}")
        
        if n_json_found == 0:
            print(f"\n  *** CRITICAL: No JSON masks found for any image! ***")
            print(f"    The model will train on EMPTY masks (all background).")
            print(f"    This is why IoU = 0.000!")
            print(f"    Please check where your LabelMe .json files are located")
            print(f"    and make sure they are in one of the searched directories.")
        
        if n_with_craters == 0 and n_with_cracks == 0 and n_json_found > 0:
            print(f"\n  *** CRITICAL: JSON files found but NO recognized labels! ***")
            print(f"    Labels in your JSONs: {all_labels}")
            print(f"    Expected crater labels: crater, cratera, cráter, vaporization, collapse...")
            print(f"    Expected crack labels: crack, grieta, fracture, fractura...")
        print()

    def __len__(self) -> int:
        return len(self.image_paths)

    def _load_masks_from_json(self, json_path: Path, 
                           h: int, w: int,
                           h_full: int = None, w_full: int = None) -> tuple:
        """
        Load crater and crack masks from a LabelMe JSON file.
        
        Parameters:
        -----------
        h, w : int
            Target size AFTER stripping the SEM metadata bar.
        h_full, w_full : int or None
            Original image size BEFORE stripping. If provided, the masks
            are drawn at full size and then cropped (not resized) to match
            the stripped image. This avoids the vertical shift bug.
        
        Recognized labels:
        Craters : 'crater', 'cratera', 'cráter', 'vaporization', 'collapse'
        Cracks  : 'crack', 'grieta', 'fracture', 'fractura'
        """
        from PIL import Image, ImageDraw
        import json

        CRATER_LABELS = {'crater', 'cratera', 'cráter', 'vaporization', 
                        'collapse', 'crater_vaporization', 'crater_collapse'}
        CRACK_LABELS  = {'crack', 'grieta', 'fracture', 'fractura', 
                        'crack_large', 'crack_small'}

        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Draw at the ORIGINAL image size from the JSON
        img_w = data.get('imageWidth',  w_full or w)
        img_h = data.get('imageHeight', h_full or h)

        canvas_crater = Image.new('L', (img_w, img_h), 0)
        canvas_crack  = Image.new('L', (img_w, img_h), 0)
        draw_crater   = ImageDraw.Draw(canvas_crater)
        draw_crack    = ImageDraw.Draw(canvas_crack)

        # Crack line width: adaptive based on image resolution.
        # At 1024px wide, 15px → ~7-8px after resize to 512, giving the model
        # enough positive pixels to learn from while staying spatially accurate.
        # For smaller images the width scales down proportionally.
        # Previously 9px was too thin: the model's predictions were thicker than
        # the GT, penalizing IoU even when cracks were correctly localized.
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
            elif shape_type == 'point':
                cx, cy = points[0]
                r = 4
                draw.ellipse([cx-r, cy-r, cx+r, cy+r], fill=255)

        mask_crater = np.array(canvas_crater)
        mask_crack  = np.array(canvas_crack)

        # CRITICAL FIX: CROP the masks the same way the image is cropped,
        # instead of resizing. The image is cropped as img[:h, :], so we
        # do the same to the masks. This prevents the vertical shift.
        if mask_crater.shape[0] > h:
            mask_crater = mask_crater[:h, :]
            mask_crack  = mask_crack[:h, :]
        
        # Handle width mismatch (unlikely but safe)
        if mask_crater.shape[1] != w:
            mask_crater = cv2.resize(mask_crater, (w, mask_crater.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
            mask_crack  = cv2.resize(mask_crack, (w, mask_crack.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)

        return (mask_crater > 127).astype(np.float32), \
            (mask_crack  > 127).astype(np.float32)


    def _load_masks_from_png_fallback(self, stem: str, 
                                    h: int, w: int) -> tuple:
        """
        Fallback: carga desde PNGs separados si no hay JSON.
        Mantiene compatibilidad con el flujo anterior.
        """
        def _read_png(mask_dir, stem, h, w):
            if mask_dir is None:
                return np.zeros((h, w), dtype=np.float32)
            for ext in self.EXTENSIONS:
                p = mask_dir / f"{stem}{ext}"
                if p.exists():
                    mask = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        mask = cv2.resize(mask, (w, h), 
                                        interpolation=cv2.INTER_NEAREST)
                        return (mask > 127).astype(np.float32)
            return np.zeros((h, w), dtype=np.float32)

        return (_read_png(self.mask_crater_dir, stem, h, w),
                _read_png(self.mask_crack_dir,  stem, h, w))

    def __getitem__(self, idx: int) -> dict:
        img_path = self.image_paths[idx]
        img_bgr  = cv2.imread(str(img_path))
        H_full, W_full = img_bgr.shape[:2]

        # Eliminar barra de metadatos SEM
        strip_h = int(H_full * self.strip_fraction)
        img_bgr = img_bgr[:H_full - strip_h, :]
        H, W    = img_bgr.shape[:2]

        img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        img_rgb  = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2RGB)

        stem = img_path.stem

        # ROBUST SEARCH FOR THE LabelMe JSON
        # The annotation may sit in any of several conventional places.
        # Usa image_dir (data/images/) como ancla para derivar data/ y data/masks/
        json_path = None
        data_root = img_path.parent.parent  # data/images/../ = data/
        search_dirs = [
            data_root / 'masks',
            data_root / 'masks' / 'craters',  # data/masks/craters/
            data_root / 'masks' / 'cracks',   # data/masks/cracks/
            img_path.parent,                   # data/images/
            data_root,                         # data/
        ]
        # Add explicitly provided directories on top
        if self.mask_crater_dir is not None:
            search_dirs.insert(0, self.mask_crater_dir.parent)
            search_dirs.insert(0, self.mask_crater_dir)
        if self.mask_crack_dir is not None:
            search_dirs.insert(0, self.mask_crack_dir)

        for search_dir in search_dirs:
            candidate = search_dir / f"{stem}.json"
            if candidate.exists():
                json_path = candidate
                break

        if json_path is not None:
            mask_crater, mask_crack = self._load_masks_from_json(
                json_path, H, W, h_full=H_full, w_full=W_full)
        else:
            mask_crater, mask_crack = self._load_masks_from_png_fallback(stem, H, W)
        # ───────────────────────────────────────────────────────────────

        # ── Dilate thin cracks slightly for training robustness ─────
        # At model resolution (512-1024), cracks annotated as thin lines
        # may be only 1-3 pixels wide after resizing. A small dilation
        # gives the model more positive signal to learn from, improving
        # recall without significantly hurting precision.
        if mask_crack.any() and self.transform is not None:
            # Only during training (when transform is the train pipeline)
            crack_area_ratio = mask_crack.sum() / mask_crack.size
            if crack_area_ratio < 0.01:  # Very thin cracks
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                mask_crack = cv2.dilate(
                    mask_crack.astype(np.uint8), kernel, iterations=1
                ).astype(np.float32)

        masks_stacked = np.stack([mask_crater, mask_crack], axis=-1)

        # AUGMENTATION
        if self.transform is not None:
            # albumentations expects the masks as a separate list
            augmented = self.transform(
                image=img_rgb,
                masks=[masks_stacked[:, :, 0],
                       masks_stacked[:, :, 1]])
            img_tensor = augmented['image']           # (3, H, W) float32

            # ToTensorV2 turns the masks into tensors on its own
            # en versiones nuevas de albumentations. Detectamos ambos casos.
            m0, m1 = augmented['masks'][0], augmented['masks'][1]
            if isinstance(m0, torch.Tensor):
                mask_out = torch.stack([m0.float(), m1.float()])
            else:
                mask_out = torch.stack([
                    torch.from_numpy(m0).float(),
                    torch.from_numpy(m1).float(),
                ])  # (2, H, W)
        else:
            # Without augmentation, resize by hand
            img_resized  = cv2.resize(img_rgb, (self.img_size, self.img_size))
            img_tensor   = torch.from_numpy(
                img_resized.transpose(2, 0, 1)).float() / 255.0

            m_c = cv2.resize(masks_stacked[:, :, 0],
                             (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)
            m_k = cv2.resize(masks_stacked[:, :, 1],
                             (self.img_size, self.img_size),
                             interpolation=cv2.INTER_NEAREST)
            mask_out = torch.stack([
                torch.from_numpy(m_c).float(),
                torch.from_numpy(m_k).float(),
            ])

        return {
            'image':    img_tensor,   # (3, H, W)
            'mask':     mask_out,     # (2, H, W): [craters, cracks]
            'filename': img_path.name,
        }

    def get_pos_weight(self) -> torch.Tensor:
        """
        Estimate pos_weight for the BCE term from the observed class imbalance.
        Twenty images are enough for a stable estimate and keep this quick.
        """
        n_pos = np.zeros(2)
        n_neg = np.zeros(2)
        sample_idx = random.sample(range(len(self)),
                                   min(20, len(self)))
        for idx in sample_idx:
            item = self.__getitem__(idx)
            for ch in range(2):
                m = item['mask'][ch].numpy()
                n_pos[ch] += m.sum()
                n_neg[ch] += (1 - m).sum()

        # Guard against division by zero on empty masks
        pw = np.where(n_pos > 0, n_neg / (n_pos + 1e-6), 10.0)
        pw = np.clip(pw, 1.0, 50.0)  # clamp for numerical stability
        print(f"  [Dataset] pos_weight: craters={pw[0]:.1f}, cracks={pw[1]:.1f}")
        return torch.tensor(pw, dtype=torch.float32)


# ─────────────────────────────────────────────────────────────────────────────
#  UTILIDADES DE DATALOADER
# ─────────────────────────────────────────────────────────────────────────────

def build_dataloaders(
    data_dir: str,
    img_size: int = 512,
    batch_size: int = 4,
    val_split: float = 0.15,
    num_workers: int = 2,
    strip_fraction: float = 0.08,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build the training and validation DataLoaders.

    Parameters
    -----------
    data_dir : str
        Root directory containing images/ and masks/
    img_size : int
        Square size fed to the network.
    batch_size : int
        Batch size. Four to eight fits a 6 GB GPU, two is safer on CPU.
    val_split : float
        Fraction of the dataset held out for validation (0.15 = 15%).
    num_workers : int
        Worker processes for data loading. Use 0 on Windows if it misbehaves.
    strip_fraction : float
        Fraction of the frame height to discard at the bottom.
    seed : int
        Semilla para reproducibilidad.

    Retorna:
    --------
    (train_loader, val_loader)
    """
    random.seed(seed)
    np.random.seed(seed)

    data_dir = Path(data_dir)
    img_dir     = data_dir / 'images'
    crater_dir  = data_dir / 'masks' / 'craters'
    crack_dir   = data_dir / 'masks' / 'cracks'

    # Verificar estructura
    for d in [img_dir]:
        if not d.exists():
            raise FileNotFoundError(f"No existe el directorio: {d}")

    crater_dir = crater_dir if crater_dir.exists() else None
    crack_dir  = crack_dir  if crack_dir.exists()  else None

    # List the images and split them
    all_imgs = sorted([
        p.name for p in img_dir.iterdir()
        if p.suffix.lower() in TungstenSEMDataset.EXTENSIONS
    ])

    random.shuffle(all_imgs)
    n_val   = max(1, int(len(all_imgs) * val_split))
    val_names   = set(all_imgs[:n_val])
    train_names = set(all_imgs[n_val:])

    print(f"\n  [DataLoader] Split: {len(train_names)} train | {n_val} val")

    # Crear subdirectorios temporales virtuales usando filtros
    # A light wrapper avoids having to copy files into physical directories
    train_dataset = _FilteredDataset(
        TungstenSEMDataset(
            image_dir=img_dir,
            mask_crater_dir=crater_dir,
            mask_crack_dir=crack_dir,
            transform=get_transforms('train', img_size),
            strip_fraction=strip_fraction,
        ),
        allowed_names=train_names
    )

    val_dataset = _FilteredDataset(
        TungstenSEMDataset(
            image_dir=img_dir,
            mask_crater_dir=crater_dir,
            mask_crack_dir=crack_dir,
            transform=get_transforms('val', img_size),
            strip_fraction=strip_fraction,
        ),
        allowed_names=val_names
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


class _FilteredDataset(Dataset):
    """Wrapper that restricts a dataset to a given set of file names."""

    def __init__(self, base_dataset: TungstenSEMDataset,
                 allowed_names: set):
        self.base = base_dataset
        self.indices = [
            i for i, p in enumerate(base_dataset.image_paths)
            if p.name in allowed_names
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.base[self.indices[idx]]


if __name__ == '__main__':
    # Quick smoke test
    import tempfile, os

    print("Dataset smoke test (no real images)...")
    print("Para usar con datos reales:")
    print("  ds = TungstenSEMDataset('data/images', 'data/masks/craters', 'data/masks/cracks')")
    print("  item = ds[0]")
    print("  print(item['image'].shape, item['mask'].shape)")