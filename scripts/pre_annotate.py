"""Generate a first pass of training masks with the classical pipeline.

This is the step that made the annotation of the dataset affordable. Labelling a
hundred SEM micrographs from scratch is a long job, so the tuned classical pipeline
was used to propose crater and crack masks, which were then corrected by hand. The
proposals are wrong often enough that they cannot be used as ground truth, but they
are close enough that correcting them is much faster than drawing from nothing.

Suggested workflow::

    1. run this script to obtain automatic proposals
    2. inspect the overlays it writes and correct them in LabelMe
    3. train the network on the corrected annotation

Usage::

    python scripts/pre_annotate.py --images ./data/images --output ./data/masks
    python scripts/pre_annotate.py --images ./data/images --output ./data/masks --scale 0.224

Outputs::

    data/masks/craters/    binary PNG per image
    data/masks/cracks/     binary PNG per image
    data/masks/overlays/   visual overlay for inspection
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import cv2


def generate_masks_from_classic(
    image_dir: str,
    output_dir: str,
    um_per_px: float = None,
    scale_override: float = None,
):
    """
    Run the classical pipeline over a directory and write mask proposals.

    Parameters
    -----------
    image_dir : str
        Directory holding the original SEM micrographs.
    output_dir : str
        Directory in which the proposed masks are written.
    um_per_px : float o None
        Scale in um/px. When None the OCR auto-detection is used.
    """
    # The classical detector supplies the proposals
    try:
        from tungsten_analyzer.classical_pipeline import (
            load_image, detect_meta_strip, auto_scale,
            compute_zones_dynamic, detect_craters, detect_cracks,
            filter_artifacts, classify_craters
        )
        print("  [pre-annotate] Classical pipeline loaded.")
    except ImportError as e:
        print(f"  [ERROR] No se puede importar tungsten_sem_analyzer: {e}")
        print("  Install the package first, for instance with: pip install -e .")
        return

    image_dir  = Path(image_dir)
    output_dir = Path(output_dir)
    crater_dir = output_dir / 'craters'
    crack_dir  = output_dir / 'cracks'
    overlay_dir = output_dir / 'overlays'

    for d in [crater_dir, crack_dir, overlay_dir]:
        d.mkdir(parents=True, exist_ok=True)

    exts = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
    imgs = sorted([p for p in image_dir.iterdir()
                   if p.suffix.lower() in exts])

    if not imgs:
        print(f"  No images found in {image_dir}")
        return

    print(f"\n  Generating proposals for {len(imgs)} images...\n")
    ok, failed = 0, 0

    for img_path in imgs:
        print(f"  Procesando: {img_path.name}")
        try:
            img_bgr, img_gray = load_image(img_path)
            H, W = img_gray.shape

            meta_h = detect_meta_strip(img_gray)
            roi_gray = img_gray[:H - meta_h, :]
            H_roi = roi_gray.shape[0]

            # Escala
            scale = scale_override or um_per_px
            if scale is None:
                # Revision 1: verified pixel size or nothing (no default value)
                from tungsten_analyzer.measurement import read_scale
                scale = read_scale(img_gray)["um_per_px"]
                if scale is None:
                    print(f"    pixel size not verified, image skipped (use --scale)")
                    continue

            # Zonas
            edge_end, artifact_start = compute_zones_dynamic(roi_gray)

            # CLASSICAL DETECTION
            empty_crack = np.zeros_like(roi_gray, dtype=bool)
            raw_craters, dark_labels = detect_craters(
                roi_gray, empty_crack, edge_end, artifact_start)
            valid_craters, _ = filter_artifacts(raw_craters, artifact_start)
            _, crater_details = classify_craters(valid_craters, scale)

            # Crater mask
            valid_labels = [c['label'] for c in crater_details]
            crater_mask = np.isin(dark_labels, valid_labels).astype(np.uint8) * 255

            # Crack mask
            all_craters_mask = crater_mask.astype(bool)
            min_skel_px = max(10, int(15.0 / scale))
            crack_mask_bool, _, _ = detect_cracks(
                roi_gray, all_craters_mask, edge_end, artifact_start,
                frangi_thresh=0.18, min_skel_px=min_skel_px,
                min_area_px=max(20, min_skel_px // 2))
            crack_mask = crack_mask_bool.astype(np.uint8) * 255

            # WRITE THE MASKS
            stem = img_path.stem

            # Rebuild a full-size mask, leaving room for the metadata bar
            crater_full = np.zeros((H, W), dtype=np.uint8)
            crater_full[:H_roi, :] = crater_mask

            crack_full = np.zeros((H, W), dtype=np.uint8)
            crack_full[:H_roi, :] = crack_mask

            cv2.imwrite(str(crater_dir / f"{stem}.png"), crater_full)
            cv2.imwrite(str(crack_dir  / f"{stem}.png"), crack_full)

            # OVERLAY FOR VISUAL INSPECTION
            overlay = img_bgr.copy()
            # Green marks craters
            overlay[:H_roi][crater_mask > 0] = (
                overlay[:H_roi][crater_mask > 0] * 0.5 +
                np.array([0, 180, 0]) * 0.5).astype(np.uint8)
            # Rojo = grietas
            overlay[:H_roi][crack_mask > 0] = (
                overlay[:H_roi][crack_mask > 0] * 0.5 +
                np.array([0, 0, 220]) * 0.5).astype(np.uint8)

            # Texto informativo
            cv2.putText(overlay, f"Verde=Craters | Rojo=Cracks | Revisa y corrige",
                        (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 255, 0), 1)
            cv2.imwrite(str(overlay_dir / f"{stem}_overlay.jpg"), overlay)

            print(f"    Craters: {np.sum(crater_mask > 0)} px | "
                  f"Grietas: {np.sum(crack_mask > 0)} px")
            ok += 1

        except Exception as e:
            print(f"    ✗ ERROR: {e}")
            failed += 1

    print(f"\n  {'─'*50}")
    print(f"  DONE: {ok} succeeded | {failed} failed")
    print(f"\n  NEXT STEP:")
    print(f"  1. Review the overlays in: {overlay_dir}/")
    print(f"  2. Correct the masks in an annotation tool")
    print(f"     (Recomendado: LabelMe https://github.com/labelmeai/labelme)")
    print(f"  3. Corrected masks must be binary PNG (0 or 255)")
    print(f"  4. Then run: python -m tungsten_analyzer.train --data-dir {output_dir.parent}")
    print(f"  {'─'*50}")


def main():
    p = argparse.ArgumentParser(
        description='Generate training mask proposals for the U-Net')
    p.add_argument('--images',  required=True,
                   help='Directory holding the SEM micrographs')
    p.add_argument('--output',  default='./data/masks',
                   help='Output directory for the masks')
    p.add_argument('--scale',   type=float, default=None,
                   help='um/px (auto-detected when omitted)')
    args = p.parse_args()

    generate_masks_from_classic(
        image_dir=args.images,
        output_dir=args.output,
        scale_override=args.scale,
    )


if __name__ == '__main__':
    main()
