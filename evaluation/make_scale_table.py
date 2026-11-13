"""Verified pixel size of every micrograph and audit of the published OCR route.

Input : results/ocr_verify_raw.csv (ocr_verify.py)
Output: results/scale_verified.csv   name, dir, export_format, mag_x, width_um, um_px
        results/ocr_route_audit.csv  per-image outcome of inference.auto_scale
        results/ocr_route_summary.csv

The verified pixel size is the field width printed in the data bar divided by the frame
width (1024 px).  It is accepted only when the magnification read from the same bar agrees
with it (width x magnification / 1024 within 110.5-112.5 um), which held for all images.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402

import numpy as np
import pandas as pd


def main():
    df = pd.read_csv(RC.RESULTS / "ocr_verify_raw.csv")
    assert df["consistent"].all(), "some data bars were not verified"
    df["um_px"] = df["width_um"] / df["image_w"]
    df["um_px_from_mag"] = 111.8 / df["mag_x"]
    df["mag_vs_width_rel"] = df["um_px_from_mag"] / df["um_px"] - 1
    cols = ["name", "dir", "export_format", "bar_rows", "mag_x", "width_um", "um_px",
            "mag_width_const", "wd_mm", "eht_kV", "signal_a", "signal_b"]
    df[cols].to_csv(RC.RESULTS / "scale_verified.csv", index=False)

    rel = df["repo_um_px"] / df["um_px"] - 1
    df["route_outcome"] = np.where(df["repo_um_px"].isna(), "failure",
                                   np.where(rel.abs() <= 0.02, "success", "silent_error"))
    df["route_um_px_effective"] = df["repo_um_px"].fillna(0.224)   # default of analyze_image
    df["route_rel_error"] = df["route_um_px_effective"] / df["um_px"] - 1
    df[["name", "dir", "export_format", "mag_x", "um_px", "repo_um_px", "route_outcome",
        "route_rel_error", "repo_meta_strip_rows"]].to_csv(RC.RESULTS / "ocr_route_audit.csv",
                                                           index=False)
    summ = (df.groupby(["dir", "export_format", "route_outcome"]).size()
              .unstack(fill_value=0).reset_index())
    summ.to_csv(RC.RESULTS / "ocr_route_summary.csv", index=False)
    print(summ.to_string(index=False))
    ok = df["route_outcome"] == "success"
    print("success rel. error: mean %.4f  max |.| %.4f"
          % (df.loc[ok, "route_rel_error"].mean(), df.loc[ok, "route_rel_error"].abs().max()))
    print("111.8/mag vs width/1024: mean %.4f  sd %.4f"
          % (df["mag_vs_width_rel"].mean(), df["mag_vs_width_rel"].std()))
    print(df.loc[~ok, ["name", "mag_x", "um_px", "repo_um_px"]].to_string(index=False))
    print(df.groupby("dir")["mag_x"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
