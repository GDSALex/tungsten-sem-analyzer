"""Redraw the training convergence figure of the paper.

Reads the per-epoch history that the training loop writes to
``checkpoints/training_history.json`` and plots the losses, the per-class IoU and the
per-class Dice, with the phase one to phase two transition marked at the epoch where
the encoder is unfrozen. Paths can be overridden from the command line, which is
useful when comparing several runs.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent.parent

_parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
_parser.add_argument('--history', default=str(_REPO_ROOT / 'checkpoints' / 'training_history.json'),
                     help='per-epoch history written by the training loop')
_parser.add_argument('--out', default=str(_REPO_ROOT / 'training_convergence.png'),
                     help='output figure')
_parser.add_argument('--freeze-epochs', type=int, default=15,
                     help='epoch at which the encoder was unfrozen, marked on the plot')
_args = _parser.parse_args()

HIST_FILE = Path(_args.history)
OUT_FILE  = Path(_args.out)

FREEZE_EPOCHS = _args.freeze_epochs

# Palette
C_TRAIN   = '#D62728'
C_VAL     = '#1F77B4'
C_CRATER  = '#FF7F0E'
C_CRACK   = '#CC0000'
C_MEAN    = '#2CA02C'
C_FREEZE  = '#8C6D31'
C_BEST    = '#E377C2'
C_GRID    = '#DDDDDD'

def load_history(path: Path):
    with open(path) as f:
        return json.load(f)

def make_plot(history, freeze_epochs, out_path):
    epochs     = [h['epoch']      for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss   = [h['val_loss']   for h in history]
    crater_iou = [h.get('crater_iou', 0.0) for h in history]
    crack_iou  = [h.get('crack_iou',  0.0) for h in history]
    mean_iou   = [h.get('mean_iou',   0.0) for h in history]
    crater_f1  = [h.get('crater_f1',  0.0) for h in history]
    crack_f1   = [h.get('crack_f1',   0.0) for h in history]
    mean_f1    = [h.get('mean_f1',    0.0) for h in history]

    scores    = [0.5 * mi + 0.5 * mf for mi, mf in zip(mean_iou, mean_f1)]
    best_idx  = int(np.argmax(scores))
    best_ep   = epochs[best_idx]
    best_miou = mean_iou[best_idx]
    best_mf1  = mean_f1[best_idx]

    plt.rcParams.update({
        'font.family':     'serif',
        'font.size':       24,
        'axes.labelsize':  28,
        'axes.titlesize':  30,
        'legend.fontsize': 22,
        'xtick.labelsize': 24,
        'ytick.labelsize': 24,
        'figure.dpi':      300,
    })

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 16), sharex=True)
    fig.patch.set_facecolor('white')

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('white')
        ax.grid(True, color=C_GRID, linewidth=1.2, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color('#888888')
            sp.set_linewidth(1.5)
        ax.tick_params(colors='black', direction='in', length=6, width=1.5)

    def vline(ax):
        if freeze_epochs < epochs[-1]:
            ax.axvline(freeze_epochs + 0.5, color=C_FREEZE, ls='--',
                       lw=3.0, alpha=0.85, zorder=3)

    # ── Panel 1: Loss (Se mantiene igual) ─────────────────────────────────────
    ax1.plot(epochs, train_loss, color=C_TRAIN,  lw=4.0, label='Train Loss', zorder=4)
    ax1.plot(epochs, val_loss,   color=C_VAL,    lw=4.0, label='Val Loss',   zorder=4)
    vline(ax1)
    ax1.set_ylabel('Loss')
    ax1.set_title('U-Net Training Convergence\nTungsten SEM Damage Segmentation',
                  fontweight='bold', pad=16)
    ax1.legend(frameon=True, framealpha=0.9, edgecolor='#BBBBBB', loc='upper right')

    # Panel 2: IoU per class
    ax2.plot(epochs, crater_iou, color=C_CRATER, lw=3.5, label='Crater IoU', zorder=4)
    ax2.plot(epochs, crack_iou,  color=C_CRACK,  lw=3.5, label='Crack IoU',  zorder=4)
    ax2.plot(epochs, mean_iou,   color=C_MEAN,   lw=5.0, label='Mean IoU',   zorder=5)
    vline(ax2)
    ax2.plot(best_ep, best_miou, marker='*', color=C_BEST, markersize=26,
             markeredgecolor='black', markeredgewidth=1.2, zorder=6)
    
    ax2.set_ylabel('IoU')
    ax2.set_ylim(-0.05, 1.05)
    # Ajuste del cuadro de leyenda
    ax2.legend(frameon=True, framealpha=0.9, edgecolor='#BBBBBB', loc='lower right',
               fontsize=18, labelspacing=0.3, handletextpad=0.5)

    # Panel 3: F1 (Dice) per class
    ax3.plot(epochs, crater_f1, color=C_CRATER, lw=3.5, ls='--', label='Crater F1', zorder=4)
    ax3.plot(epochs, crack_f1,  color=C_CRACK,  lw=3.5, ls='--', label='Crack F1',  zorder=4)
    ax3.plot(epochs, mean_f1,   color=C_MEAN,   lw=5.0,          label='Mean F1',   zorder=5)
    vline(ax3)
    ax3.plot(best_ep, best_mf1, marker='*', color=C_BEST, markersize=26,
             markeredgecolor='black', markeredgewidth=1.2, zorder=6)

    ax3.set_xlabel('Epoch', fontweight='bold')
    ax3.set_ylabel('F1 (Dice)')
    ax3.set_ylim(-0.05, 1.05)
    # Ajuste del cuadro de leyenda
    ax3.legend(frameon=True, framealpha=0.9, edgecolor='#BBBBBB', loc='lower right',
               fontsize=18, labelspacing=0.3, handletextpad=0.5)

    fig.subplots_adjust(hspace=0.15)
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {out_path}")

if __name__ == '__main__':
    history = load_history(HIST_FILE)
    make_plot(history, FREEZE_EPOCHS, OUT_FILE)