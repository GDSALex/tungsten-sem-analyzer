"""Recompute the crack regularity classification of Section 8.7.

A crack network counts as spatially regular when at least three cracks reaching the
surface are spaced with a coefficient of variation below 0.45. The idea behind the
metric is that once a crack forms it relieves the in-plane stress over an exclusion
zone proportional to its length, so on subsequent thermal cycles new cracks nucleate
preferentially at comparable distances, and a periodic pattern builds up. Regularity
therefore points to fatigue-driven organisation, whereas an irregular network points
to nucleation controlled by pre-existing defects.

This script rereads the JSON reports written by the inference workflow, which is far
quicker than re-running inference over the whole campaign. Working from the reports
means the surface position of each crack is taken from its centroid rather than from
the topmost pixel of its skeleton, which is an approximation, but a much closer one
than the criterion it replaces.

Usage::

    python scripts/crack_regularity.py --dir ./inference_results
    python scripts/crack_regularity.py --dir ./inference_ATW_results
"""

import json
import argparse
import numpy as np
from pathlib import Path


def recompute_regularity(crack_details, edge_end, image_height):
    """
    Improved regularity algorithm applied to existing JSON crack data.
    Uses centroid[1] as approximation for surface X position.
    """
    surface_margin = max(40, int(0.08 * image_height))

    near_surface = [
        c for c in crack_details
        if c['connects_edge'] or c['bbox'][0] < (edge_end + surface_margin)
    ]

    regular_cls = 0
    regular_cv = None

    if len(near_surface) >= 3:
        # Use centroid X as approximation for surface X position
        # (topmost pixel X requires the actual segmentation mask)
        surface_x = sorted(c['centroid'][1] for c in near_surface)
        spacings = np.diff(surface_x)
        mean_sp = float(np.mean(spacings))

        if mean_sp > 1:
            cv_val = float(np.std(spacings) / (mean_sp + 1e-9))
            regular_cv = round(cv_val, 3)
            if cv_val < 0.45:
                regular_cls = 1

    return regular_cls, regular_cv


def process_directory(results_dir):
    results_dir = Path(results_dir)
    json_files = sorted(results_dir.glob('*_report.json'))

    if not json_files:
        print(f"No *_report.json files found in {results_dir}")
        return

    changed = 0
    total = 0

    for jf in json_files:
        with open(jf, 'r', encoding='utf-8') as f:
            report = json.load(f)

        crack_details = report.get('crack_details', [])
        edge_end = report.get('zones', {}).get('edge_end_px', 0)
        image_height = report.get('dimensions', {}).get('height_px', 768)

        old_regular = report['cracks']['regular_crack_class']
        new_regular, new_cv = recompute_regularity(crack_details, edge_end, image_height)

        report['cracks']['regular_crack_class'] = new_regular
        report['cracks']['regular_crack_class_desc'] = 'Yes (CV < 45%)' if new_regular else 'No'
        report['cracks']['regular_cv'] = new_cv

        total += 1
        if new_regular != old_regular:
            changed += 1
            status = "REGULAR" if new_regular else "IRREGULAR"
            cv_str = f"CV={new_cv:.3f}" if new_cv is not None else "n/a"
            print(f"  CHANGED  {jf.name[:45]:45s}  {status}  ({cv_str})")

        with open(jf, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)

    print(f"\n{'-'*60}")
    print(f"  Processed: {total} files")
    print(f"  Changed:   {changed} files")
    print(f"  Now regular: {sum(1 for jf in json_files if json.load(open(jf))['cracks']['regular_crack_class'] == 1)}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dir', default='./inference_results',
                   help='Directory containing *_report.json files')
    args = p.parse_args()

    print(f"\n{'='*60}")
    print(f"  REGULARITY RECOMPUTE — improved algorithm (CV < 0.45)")
    print(f"{'='*60}\n")
    process_directory(args.dir)
