"""Automated quantification of high heat flux damage in tungsten SEM micrographs.

This package accompanies the OLMAT post-exposure analysis paper and holds the three
stages it describes.

:mod:`classical_pipeline`
    The handcrafted computer vision detector, tuned by evolutionary search. It
    produced the pre-annotation that made labelling the dataset feasible, and its
    topographic edge scanner is still used inside the deep-learning workflow.

:mod:`genetic_search`
    The NSGA-II search over its forty parameters.

:mod:`model`, :mod:`dataset`, :mod:`train`
    The U-Net with a ResNet34 encoder, the data pipeline that feeds it and the
    two-phase training loop with progressive unfreezing.

:mod:`inference`
    The multi-scale inference workflow that turns a micrograph into physical damage
    metrics, in micrometres, with the scale read from the image itself.

:mod:`metrics`
    The evaluation protocol. Every number reported in the paper comes from here, and
    both methods are scored by the same rules.

:mod:`validation_figures`
    The per-image comparison figures against the manual annotation.
"""

__version__ = "1.0.0"

__all__ = [
    "classical_pipeline",
    "genetic_search",
    "model",
    "dataset",
    "train",
    "inference",
    "metrics",
    "validation_figures",
]
