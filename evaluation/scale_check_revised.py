"""Check the scale reader of the revised code (tungsten_analyzer/measurement.py)
against the verified pixel sizes of results/scale_verified.csv, on all 212 images.

For each image the revised reader either returns a pixel size (accepted only when the
magnification and the field width it reads agree within 1 %) or withholds it.  A returned
value is wrong when it differs from the verified one by more than 1 %.
Output: results/scale_check_revised.csv and a printed summary.
"""
from __future__ import annotations

import importlib.util
import sys
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402

import pandas as pd

MEAS = RC.REPO / "tungsten_analyzer" / "measurement.py"


def _load_measurement():
    spec = importlib.util.spec_from_file_location("measurement_revised", MEAS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    import pytesseract
    cmd = mod.locate_tesseract()
    if cmd:
        pytesseract.pytesseract.tesseract_cmd = cmd
    return mod


def check(path_str: str) -> dict:
    RC.set_low_priority_off()
    ms = _load_measurement()
    p = Path(path_str)
    res = ms.read_scale(RC.read_gray(p))
    return {"name": p.stem, "dir": p.parent.name, "um_px_tool": res["um_per_px"],
            "verified_by_tool": res["verified"], "mag_read": res["magnification"],
            "width_read_um": res["width_um"], "reason": res["reason"]}


def main():
    paths = [str(p) for p in RC.list_images(RC.IMG_DIR)] + [str(p) for p in RC.list_images(RC.ATW_DIR)]
    with Pool(4, initializer=RC.set_low_priority_off) as pool:
        recs = pool.map(check, paths)
    df = pd.DataFrame(recs)
    ref = pd.read_csv(RC.RESULTS / "scale_verified.csv")[["name", "um_px"]]
    df = df.merge(ref, on="name", how="left")
    df["rel_err"] = df["um_px_tool"] / df["um_px"] - 1.0
    df["outcome"] = "withheld"
    df.loc[df.um_px_tool.notna() & (df.rel_err.abs() <= 0.01), "outcome"] = "correct"
    df.loc[df.um_px_tool.notna() & (df.rel_err.abs() > 0.01), "outcome"] = "wrong"
    df.to_csv(RC.RESULTS / "scale_check_revised.csv", index=False)
    print(df.groupby(["dir", "outcome"]).size().to_string())
    print("max |rel err| of returned values:", float(df.rel_err.abs().max()))
    w = df[df.outcome != "correct"]
    if len(w):
        print(w[["name", "mag_read", "width_read_um", "reason", "um_px"]].to_string(index=False))


if __name__ == "__main__":
    main()
