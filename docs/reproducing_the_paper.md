# Reproducing the results of the paper

This walks through every table and figure of the manuscript and says which command
produces it. It assumes the package is installed (`pip install -e .`) and that the
dataset sits under `./data` as described in `data/README.md`.

Two things are worth knowing before starting.

The trained weights are not in the repository, since GitHub is a poor home for a
300 MB checkpoint. `best_model.pth` is available from the corresponding author on
reasonable request, together with the annotated dataset (see `data/README.md`);
place it under `checkpoints/`.

Everything reported in the first version of the paper comes from the checkpoint of
epoch 125, selected on the 14 held-out images, and the commands below reproduce that
version. The revised manuscript (revision 1) does not use this checkpoint for any of
its results: it reports out-of-fold results of networks retrained inside
cross-validation grouped by exposed sample edge, computed by the evaluation code
archived with the revision. Its Section S7.1 explains how the Table 1 of the first
version relates to the revised figures. The train and held-out subsets are recovered by replaying the
split of the training run, a plain random shuffle of the sorted file names under seed
42 with the first fifteen per cent held out. Change that seed and the subsets stop
corresponding to what the checkpoint actually saw during training.

---

## Table 1 of the first version, segmentation performance (superseded)

```bash
python scripts/reproduce_table1.py \
    --data-dir ./data \
    --checkpoint checkpoints/best_model.pth \
    --thr-crater 0.5 --thr-crack 0.30 \
    --save table1_records.json
```

The printed table was produced with the crack channel thresholded at 0.30, the default
of `predict_image`, although the text of the first version stated 0.25; the two
thresholds are now explicit arguments, and `--thr-crack 0.25` gives a crack score of
0.731 instead of 0.740.

Roughly fifteen minutes on a GPU, most of it spent on the classical pipeline. Add
`--skip-classical` to evaluate only the network, which takes about two.

The script prints the medians and the means for the whole annotated set and for the
training and held-out subsets separately, together with the image-level detection and
false alarm counts discussed in Section 7.2 of the first version. With `--save` it
also writes the per-image records, which is what you want if you intend to plot
distributions or inspect individual failures.

Expected output, from the checkpoint of the first version. The crack columns are the
legacy score of that version, which is not an IoU (it is defined in Section S5.1 of
the supplementary material of the revised manuscript), and the F1 derived from it as
2L/(1 + L); they are in-sample figures, since 85 of the 99 images were used for
training and the other 14 for checkpoint selection.

| | craters IoU | craters F1 | cracks, legacy score L | cracks, 2L/(1 + L) |
| --- | --- | --- | --- | --- |
| Classical, median | 0.264 | 0.418 | 0.000 | 0.000 |
| U-Net, median | 0.839 | 0.912 | 0.740 | 0.850 |
| U-Net, training subset | 0.845 | 0.916 | 0.743 | 0.853 |
| U-Net, held-out subset | 0.780 | 0.876 | 0.536 | 0.698 |

The zero median of the classical pipeline on cracks is not an artefact of the
protocol. That detector returns no crack at all on 33 of the 71 annotated
micrographs, and on a further 13 what it returns falls somewhere other than the
annotation. It is also the reason the network was built.

---

## Figure 3 of the first version, training log (Figure S1 of the revised supplementary material)

```bash
python scripts/plot_convergence.py
```

Reads `checkpoints/training_history.json`, which the training loop writes as it goes,
and redraws the loss and metric curves with the styling used in the paper. The phase
one to phase two transition, where the encoder is unfrozen, is marked.

## Figure 4 of the first version, distribution of the per-image metrics

The histograms are built from the per-image records written by
`reproduce_table1.py --save`. Each panel bins the per-image scores of one metric over
twenty equal-width bins in [0, 1], with the mean and the median marked.

## Figures 5 to 7 of the first version, per-image comparison against the annotation

```bash
python -m tungsten_analyzer.validation_figures \
    --checkpoint checkpoints/best_model.pth \
    --data-dir ./data \
    --output-dir ./validation_results \
    --num-images 0
```

`--num-images 0` processes the whole set; give it a number to sample. Each figure
has two rows, craters above and cracks below, and four columns: the original
micrograph, the annotation, the probability map and an error overlay in which green
marks true positives, red false positives and blue false negatives.

Note which images belong to which subset. Of the figures of the first version,
ALMW1-3a and ALMW1-1c are held out from training, while ALMW3-1a and ALMW3-2d are
training images and are captioned as such. The crack panels now show the strict IoU
and F1, computed from the same pixel counts, with the legacy score shown separately.
The error figures of the revised manuscript are out-of-fold predictions drawn by the
evaluation code of the revision.

## Figures 8 to 14 of the first version, full inference output

```bash
python -m tungsten_analyzer.inference \
    --checkpoint checkpoints/best_model.pth \
    --input ./data/images \
    --output ./inference_results
```

Writes, for every micrograph, an annotated segmentation figure and a JSON report with
the physical metrics: crater count, equivalent diameters, total and largest area in
square micrometres, the size category of the largest region (a size class only, which
replaces the severity grades of the first version), crack count split into edge
connected, near edge and internal, total crack length in micrometres, and the
regularity classification. The pixel size and its source are written into every
report, and the physical outputs of an image are withheld when its data bar cannot be
read and verified, unless the pixel size is given explicitly.

Point `--input` at the ATW directory to process the ATW images, which the revised
manuscript uses as a qualitative external application (the first version presented
them as an independent validation in Section 7.4).

## Crack network regularity (Section 8.7 of the first version, withdrawn in revision 1)

```bash
python scripts/crack_regularity.py --dir ./inference_results
python scripts/crack_regularity.py --dir ./inference_ATW_results
```

Recomputes the regularity classification over the JSON reports without re-running
inference. A network counts as spatially regular when at least three cracks reaching
the surface have an inter-crack spacing with a coefficient of variation below 0.45.

---

## Retraining from scratch

Not needed to reproduce the tables, but the path is short.

```bash
# 1. mask proposals from the classical pipeline, then corrected by hand in LabelMe
python scripts/pre_annotate.py --images ./data/images --output ./data/masks

# 2. training, roughly four hours on a single mid-range GPU
python -m tungsten_analyzer.train --data-dir ./data --epochs 180 --batch-size 4
```

Expect small numerical differences from the published figures. Augmentation and the
data loader shuffle are seeded per run, so a retrained model will not be bitwise
identical to the released checkpoint.

## Re-running the evolutionary search

```bash
python -m tungsten_analyzer.genetic_search --dataset ./annotation_subset --trials 300
```

Only necessary to reproduce the search itself, since its result is already the
`CFG` dictionary of `classical_pipeline.py`. The search was run three times on the same
78 images, and the run that produced `CFG` completed its budget of 300 evaluations; the
first version stated, incorrectly, that it stopped at evaluation 261. The record of the
three runs is in Section S2.4 of the supplementary material of the revised manuscript.
NSGA-II is stochastic and a fresh run will
land on a different but comparable parameter set.
