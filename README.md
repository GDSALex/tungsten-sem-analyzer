# TungstenAnalyzer

Automated detection and quantification of high heat flux damage in scanning electron
microscopy images of tungsten, developed for post-exposure analysis of the samples
irradiated at the [OLMAT](https://www.ciemat.es) facility of CIEMAT.

This repository holds the code behind the paper *A first deep learning framework for
the post-exposure quantification of high heat flux damage in tungsten at OLMAT*, and
it is complete enough to reproduce the results reported there from the annotated
dataset and the trained weights.

---

## What the tool does

Tungsten plasma-facing components exposed to disruption-like thermal loads develop
two damage morphologies that dominate at the scales SEM resolves: thermal fatigue
crack networks, and crater-like features left by ablation, melting and droplet
ejection. Measuring them by hand is slow, and worse, it is subjective enough that
two operators do not produce comparable numbers.

The tool segments both morphologies and converts them into physical quantities in
micrometres, with the scale read automatically from the metadata bar of the
micrograph. For each image it reports the number of craters, their equivalent
diameters and total damaged area, a five-level severity grade, the number and total
length of cracks, whether each crack reaches the material edge, and a regularity
metric derived from the coefficient of variation of the inter-crack spacing.

It is a measurement layer, not a damage model. It does not attempt any
thermomechanical evaluation, any reconstruction of the surface temperature or any
thermal fatigue assessment. Its purpose is to produce objective numbers for a whole
campaign, so that the small number of exposures worth a slower and considerably more
expensive characterisation can be chosen on evidence rather than by eye.

---

## How it is organised

The work proceeded in three stages, and the code still follows that shape.

| Module | What it is |
| --- | --- |
| `tungsten_analyzer/classical_pipeline.py` | Handcrafted detector: topographic edge scanner, multichannel watershed for craters, Frangi and black-hat ridge detection for cracks. |
| `tungsten_analyzer/genetic_search.py` | NSGA-II search over its forty free parameters, through Optuna. |
| `tungsten_analyzer/model.py` | U-Net with a ResNet34 encoder, scSE attention gates, deep supervision, and the composite loss. |
| `tungsten_analyzer/dataset.py` | Dataset and the SEM-specific augmentation chain. Reads LabelMe annotations directly. |
| `tungsten_analyzer/train.py` | Two-phase training loop with progressive unfreezing of the encoder. |
| `tungsten_analyzer/inference.py` | Multi-scale inference and the extraction of physical metrics. |
| `tungsten_analyzer/metrics.py` | The evaluation protocol. Every reported number comes from here. |
| `tungsten_analyzer/validation_figures.py` | Per-image comparison figures against the annotation. |

The desktop application described in Appendix B of the paper is not part of this
release; `app/README.md` explains why and what reaches the same result from the
command line.

The classical pipeline was not discarded when the network arrived. It generated the
pre-annotation that made labelling the dataset affordable, and its edge scanner still
runs inside the deep-learning workflow, where it supplies the physical frame of
reference in which the network output is interpreted: craters above the material edge
are discarded, and cracks are classified as edge-connected or internal according to
whether they reach the edge zone.

---

## Installation

Python 3.9 or newer, and a CUDA-capable GPU if you intend to train. Inference and
evaluation run on CPU, more slowly.

```bash
git clone https://github.com/<user>/tungsten-sem-analyzer.git
cd tungsten-sem-analyzer
python -m venv .venv && source .venv/bin/activate     # .venv\Scripts\activate on Windows
pip install -e .
```

Reading the scale from the metadata bar needs the Tesseract OCR binary installed on
the system, not only the `pytesseract` wrapper. Without it, pass the scale explicitly
with `--scale`.

---

## Data layout

The images themselves are not redistributed here. See `data/README.md` for the
expected layout and for how to obtain them.

```
data/
  images/    SEM micrographs (.jpg, .png, .tif)
  masks/     one LabelMe .json per image, same file stem
```

Annotations are read straight from the LabelMe JSON rather than from pre-rendered
masks, which keeps one single source of truth. Craters are closed polygons. Cracks
are open polylines following the centreline of the fracture, rasterised at a fixed
width of 15 pixels for a 1024-pixel-wide frame. That width is a drawing convention,
not a measurement, and the same convention is applied when building training targets
and when evaluating, so the two never disagree.

---

## Reproducing the paper

```bash
# Table 1, both methods, one evaluation protocol
python scripts/reproduce_table1.py --data-dir ./data --checkpoint checkpoints/best_model.pth

# Figure 3, training convergence
python scripts/plot_convergence.py

# Figures 5 to 7, per-image comparison against the annotation
python -m tungsten_analyzer.validation_figures --checkpoint checkpoints/best_model.pth \
    --data-dir ./data --num-images 0

# Crack regularity statistics of Section 8.7
python scripts/crack_regularity.py --dir ./inference_results
```

`docs/reproducing_the_paper.md` walks through this in more detail, including which
figure and table each command produces.

## Training from scratch

```bash
# 1. propose masks with the classical pipeline, then correct them in LabelMe
python scripts/pre_annotate.py --images ./data/images --output ./data/masks

# 2. train
python -m tungsten_analyzer.train --data-dir ./data --epochs 160 --batch-size 2

# 3. run the full inference workflow over a directory
python -m tungsten_analyzer.inference --input ./data/images --output ./inference_results
```

Re-running the evolutionary search is only necessary to reproduce the search itself;
its result is already hard-coded in `classical_pipeline.CFG`.

```bash
python -m tungsten_analyzer.genetic_search --dataset ./annotation_subset --trials 300
```

---

## A note on the evaluation protocol

The comparison between the two methods is easy to get wrong, so it is worth being
explicit about what `metrics.py` does.

Crater masks are compared directly, with no tolerance. Crack masks are not, for two
reasons. The annotation is a centreline and therefore carries no width information,
so a strict pixel overlap would largely measure how faithfully a method reproduces an
arbitrary rasterisation convention. And the two methods return lines of very
different thickness, fifteen pixels for the network against three for the classical
detector, which on its own would account for a large part of the difference between
them.

Both effects are removed. Each crack prediction is reduced to its skeleton and
redrawn at the width of the annotation, and both masks are then dilated by five
pixels before the overlap is counted. What survives is a measure of whether the crack
was found and whether it sits where the annotator drew it, which is also what matters
downstream, since crack length and density are computed from the skeleton.

Images are scored one by one. An image enters the statistic of a class only when it
carries an annotation of that class, because an overlap score is undefined against an
empty reference. Nothing else is filtered: images where a method finds nothing are
kept and enter as zeros. Since that exclusion hides what happens on clean surfaces,
`metrics.image_level_counts` reports detection and false alarm rates per image
separately.

---

## Citation

If this code is useful in your work, please cite the paper. `CITATION.cff` holds the
machine-readable entry; GitHub renders it in the sidebar.

## Licence

MIT, see `LICENSE`. The SEM micrographs are not covered by it and remain subject to
the data policy of the OLMAT facility.

## Acknowledgements

Developed at the Laboratorio Nacional de Fusión, CIEMAT, within the framework of the
Máster de Energía y Combustibles del Futuro of the Universidad Autónoma de Madrid.
This work has been partially carried out within the framework of the EUROfusion
Consortium, funded by the European Union via the Euratom Research and Training
Programme (Grant Agreement No 101052200 — EUROfusion). The views and opinions
expressed are those of the authors only and do not necessarily reflect those of the
European Union or the European Commission.
