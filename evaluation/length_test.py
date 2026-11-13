"""Synthetic orientation test of the crack-length estimators (reviewer comment 6).

Straight cracks of known length are drawn at every orientation between 0 and 90 degrees,
either as one-pixel digital lines (pure estimator bias) or as bands of the width that the
network produces (skeletonisation plus estimator), and their length is measured with the
pixel count of the submitted manuscript, the Freeman weights (1, sqrt 2) and the Kulpa /
Dorst-Smeulders weights (0.948, 1.343).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from skimage.morphology import skeletonize

import eval_lib as EL


def draw(L, theta_deg, width, size=1024):
    img = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(img)
    c = size / 2
    dx, dy = np.cos(np.radians(theta_deg)) * L / 2, np.sin(np.radians(theta_deg)) * L / 2
    d.line([(c - dx, c - dy), (c + dx, c + dy)], fill=255, width=width)
    return np.array(img) > 127


def main():
    rows = []
    for L in (50, 100, 200, 400):
        for w in (1, 7, 15):
            for th in np.arange(0, 90.01, 1.0):
                m = draw(L, th, w)
                sk = m if w == 1 else skeletonize(m)
                for meth in ("count", "freeman", "kulpa"):
                    rows.append({"L": L, "width": w, "theta": th, "method": meth,
                                 "est": EL.skeleton_length(sk, meth)})
                # revised estimator of the revision: holes filled, spurs pruned, Kulpa weights
                rs = m if w == 1 else EL.crack_skeleton(m)
                rows.append({"L": L, "width": w, "theta": th, "method": "revised",
                             "est": EL.skeleton_length(rs, "kulpa")})
    df = pd.DataFrame(rows)
    df["rel_err"] = df["est"] / df["L"] - 1
    df.to_csv(RC.RESULTS / "length_test.csv", index=False)
    s = (df.groupby(["width", "L", "method"])["rel_err"]
           .agg(mean="mean", min="min", max="max").mul(100).round(1))
    print(s.to_string())


if __name__ == "__main__":
    main()
