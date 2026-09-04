# Dataset

The SEM micrographs are not redistributed in this repository. They are available from
the corresponding author on reasonable request, and their use is subject to the data
policy of the OLMAT facility.

## Expected layout

```
data/
  images/    SEM micrographs, one file per field of view (.jpg, .png, .tif)
  masks/     one LabelMe annotation per image, same file stem, .json
```

The stem must match. `ALMW1-3a-600MW_05.jpg` is paired with
`ALMW1-3a-600MW_05.json`, and an image without its JSON is skipped.

## Naming convention

The file name encodes the exposure, which is how the analysis scripts recover the
experimental conditions without a separate table:

```
ALMW3-4d-600MW-4p-HT_01.jpg
  |     |    |    |  |   |
  |     |    |    |  |   +-- field of view on that position
  |     |    |    |  +------ pre-heated above the DBTT (absent means room temperature)
  |     |    |    +--------- number of pulses (absent means one)
  |     |    +-------------- absorbed power density in MW/m2
  |     +------------------- sample number and exposure position
  +------------------------- batch and campaign
```

## Annotation format

Annotations are made in [LabelMe](https://github.com/labelmeai/labelme) and are read
directly from the JSON, never from pre-rendered masks, so that there is one single
source of truth for the ground truth.

Two labels are used.

`crater`, drawn as a closed **polygon** around the boundary of the melted, ablated or
collapsed region. Craters have a real, measurable boundary, so the polygon is a
measurement and the masks are compared directly, with no tolerance.

`crack`, drawn as an open **polyline** following the centreline of the fracture. At
these magnifications a thermal fatigue crack is one to two pixels across and its
apparent width shifts with the local topographic contrast of the microscope, so the
edges are not something two annotators would agree on. The centreline is. The
polyline is rasterised at a fixed width of 15 pixels for a 1024-pixel-wide frame,
scaled proportionally for other sizes; that width is a drawing convention, and it is
applied identically when building training targets and when evaluating.

The loader tolerates a few alternative spellings of both labels, listed in
`tungsten_analyzer/metrics.py`, but the dataset of the paper uses only these two.

## Metadata bar

Every frame carries the SEM metadata bar along the bottom. It is cropped away before
anything else happens, by default the lower 8 per cent of the frame, and the
annotation is cropped to match. The bar is read first, by OCR, to recover the scale
in micrometres per pixel; when Tesseract is unavailable the scale has to be supplied
with `--scale`.

## The dataset of the paper

99 annotated micrographs of ITER-grade tungsten from the ALMW campaign, 1024 by 768
pixels, covering absorbed power densities from 300 to 1200 MW/m2 with 1, 2, 4 and 7
pulses of 2 ms, seventeen of them from samples pre-heated to between 500 and 600 °C.
Of the 99, 93 carry at least one crater annotation, 71 carry at least one crack
annotation and 4 carry none at all, being fields of view from undamaged regions.

A second campaign, the ATW series, contributes 113 further micrographs from a
different tungsten supplier. They are not annotated and were never used for training
or threshold selection, which is what makes them an independent check of
generalisation.
