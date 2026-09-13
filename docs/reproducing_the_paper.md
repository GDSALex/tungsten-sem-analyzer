# Reproducing the results of the paper

This walks through every table and figure of the manuscript and says which command
produces it. It assumes the package is installed (`pip install -e .`) and that the
dataset sits under `./data` as described in `data/README.md`.

Two things are worth knowing before starting.

The trained weights are not in the repository, since GitHub is a poor home for a
300 MB checkpoint. `best_model.pth` is available from the corresponding author on
reasonable request, together with the annotated dataset (see `data/README.md`);
place it under `checkpoints/`.

Everything reported in the paper comes from the checkpoint of epoch 125, selected on
the validation subset. The train and held-out subsets are recovered by replaying the
split of the training run, a plain random shuffle of the sorted file names under seed
42 with the first fifteen per cent held out. Change that seed and the subsets stop
corresponding to what the checkpoint actually saw during training.

---

## Table 1, segmentation performance

```bash
python scripts/reproduce_table1.py \
    --data-dir ./data \
    --checkpoint checkpoints/best_model.pth \
    --save table1_records.json
```

Roughly fifteen minutes on a GPU, most of it spent on the classical pipeline. Add
`--skip-classical` to evaluate only the network, which takes about two.

The script prints the medians and the means for the whole annotated set and for the
training and held-out subsets separately, together with the image-level detection and
false alarm counts discussed in Section 7.2. With `--save` it also writes the
per-image records, which is what you want if you intend to plot distributions or
inspect individual failures.

Expected output, from the checkpoint of the paper:

| | craters IoU | craters F1 | cracks IoU | cracks F1 |
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

## Figure 3, training convergence

```bash
python scripts/plot_convergence.py
```

Reads `checkpoints/training_history.json`, which the training loop writes as it goes,
and redraws the loss and metric curves with the styling used in the paper. The phase
one to phase two transition, where the encoder is unfrozen, is marked.

## Figure 4, distribution of the per-image metrics

The histograms are built from the per-image records written by
`reproduce_table1.py --save`. Each panel bins the per-image scores of one metric over
twenty equal-width bins in [0, 1], with the mean and the median marked.

## Figures 5 to 7, per-image comparison against the annotation

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

Note which images belong to which subset. Of the figures in the paper, ALMW1-3a and
ALMW1-1c are held out from training, while ALMW3-1a and ALMW3-2d are training images
and are captioned as such.

## Figures 8 to 14, full inference output

```bash
python -m tungsten_analyzer.inference \
    --checkpoint checkpoints/best_model.pth \
    --input ./data/images \
    --output ./inference_results
```

Writes, for every micrograph, an annotated segmentation figure and a JSON report with
the physical metrics: crater count, grades, equivalent diameters and total area in
square micrometres, crack count split into edge-connected and internal, total crack
length in micrometres, and the regularity classification.

Point `--input` at the ATW directory to reproduce the independent validation of
Section 7.4.

## Section 8.7, crack network regularity

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
python -m tungsten_analyzer.train --data-dir ./data --epochs 160 --batch-size 2
```

Expect small numerical differences from the published figures. Augmentation and the
data loader shuffle are seeded per run, so a retrained model will not be bitwise
identical to the released checkpoint.

## Re-running the evolutionary search

```bash
python -m tungsten_analyzer.genetic_search --dataset ./annotation_subset --trials 300
```

Only necessary to reproduce the search itself, since its result is already the
`CFG` dictionary of `classical_pipeline.py`. The run reported in the paper was
stopped at evaluation 261, by which point the best composite score had improved by
less than 0.01 over the preceding seventy. NSGA-II is stochastic and a fresh run will
land on a different but comparable parameter set.
