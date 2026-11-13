# Evaluation code of revision 1

These scripts compute the numbers, tables and figures of Section 4 of the revised manuscript
(ANUCENE-D-26-01513, revision 1) from per-image record files, and contain the grouped
cross-validation, the architecture comparison and the checks of the pixel size and of the
measurements that are described there. They import the model, the training routine, the
dataset, the metrics and the analysis stage from the repository root, so the code that is
evaluated is the code of the tool.

What is **not** in the repository: the SEM micrographs, the LabelMe annotations and the trained
weights, which are subject to the data policy of the OLMAT facility and are available from the
corresponding author on reasonable request. The split manifest, the per-image evaluation records
and the training configurations are provided as supplementary data files of the paper, which are
the inputs of `numbers_v2.py` and `aggregate.py`.

## Where the files are looked for

Paths are set by environment variables; nothing is read from an absolute path.

| Variable | Content | Default |
|---|---|---|
| `TUNGSTEN_WORK` | output tree: `results/`, `cv_runs/`, `arch_runs/`, `figures/` | `evaluation/work` |
| `TUNGSTEN_DATA` | `images/` (JPEG micrographs) and `masks/` (LabelMe JSON) of the 99 ALMW images | `<WORK>/data` |
| `TUNGSTEN_ATW` | the 113 ATW micrographs | `<DATA>/atw` |
| `TUNGSTEN_CKPT` | `best_model.pth`, the checkpoint of the first submission | `<WORK>/checkpoints` |
| `TESSERACT_CMD` | Tesseract binary, for the scale checks | found on the `PATH` |

Run the scripts from this folder with the packages of `requirements.txt` plus `pandas`, `scipy`,
`scikit-image`, `matplotlib` and `pytesseract`. The numbers of the paper were computed with
Python 3.10, numpy 1.26.4, torch 2.3.0 (CUDA 12.1), segmentation-models-pytorch 0.5.0,
albumentations 2.0.8, opencv-python 4.13 and scikit-image 0.23.2.

## Order of the computations

| Step | Script | Writes |
|---|---|---|
| Pixel size of every image | `ocr_verify.py`, `make_scale_table.py`, `scale_check_revised.py` | `results/scale_verified.csv` and checks of the scale reader |
| Edge-grouped and leave-one-specimen-out cross-validation | `train_cv.py`, `run_queue.py`, `evaluate.py` | `cv_runs/`, `results/cv_records.csv` |
| Comparison of network configurations | `arch_train.py`, `arch_queue.py`, `arch_eval.py` | `arch_runs/`, `results/arch_*.csv` |
| Adopted network: final model and per-image records | `eval_final.py maps <arch>`, `eval_final.py eval <arch>` | `results/v2/*.csv` |
| Complements | `bridging_check.py`, `atw_measure.py`, `length_test.py`, `common_loss_published.py`, `timing.py` | `results/*.csv` |
| Numbers and table bodies of the manuscript | `numbers_v2.py <adopted> <reference>` (and `make_numbers.py` for the quantities carried over from the first version) | `results/v2/numbers.json`, `results/v2/tex/` |
| Figures | `make_figs_v2.py <arch>`, `fig_hybrid.py`, `fig_convergence.py`, `fig_distributions.py`, `fig_examples.py`, `fig_failures.py` | `figures/` |
| Summary tables | `aggregate.py` | `results/summary_*.csv` |

`numbers_v2.py` also checks every fixed claim of the text against the values and prints
`CLAIM NOT SUPPORTED` when one is no longer true. `arch_eval.py` contains the decision rule for
the comparison of configurations, written before any result of the comparison was seen.

The comparison of configurations was stopped early: U-Net++ and the ConvNeXt encoder were not
completed and are not reported as results (see the supplementary material of the paper).
