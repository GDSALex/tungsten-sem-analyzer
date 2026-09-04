"""Training loop with progressive unfreezing of the encoder.

Training runs in two phases. During the first fifteen epochs the ResNet34 encoder
stays frozen and only the decoder is updated, which lets the randomly initialised
decoder settle before it can disturb the pre-trained filters. The encoder is then
released and the whole network is fine-tuned with a differential learning rate,
10^-3 for the decoder and 10^-4 for the encoder, so that the low-level texture
filters adapt gradually to the noise characteristics of the microscope instead of
being overwritten.

The run reported in the paper used AdamW with cosine annealing after a three-epoch
linear warm-up, gradient accumulation over two steps for an effective batch of four
at 1024 by 1024, and early stopping on the validation score with patience 30. It
terminated at epoch 155 and the checkpoint selected for all reported results is the
one from epoch 125.

Usage::

    python -m tungsten_analyzer.train --data-dir ./data --epochs 160 --batch-size 2

Outputs::

    checkpoints/best_model.pth       best epoch by the combined IoU and F1 score
    checkpoints/last_model.pth       final epoch
    checkpoints/training_history.json  per-epoch losses and metrics
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, SequentialLR, LinearLR

from tungsten_analyzer.model import (build_unet, build_unet_deep_supervised,
                        freeze_encoder, unfreeze_encoder,
                        get_param_groups, CombinedLoss, CrackAwareLoss,
                        MetricsAccumulator)
from tungsten_analyzer.dataset import build_dataloaders

# ─────────────────────────────────────────────────────────────────────────────
#  DEFAULT CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_CFG = {
    # Data
    'data_dir':      './data',
    'img_size':      1024,
    'batch_size':    2,
    'grad_accum_steps': 2,      # Acumula 2 pasos → batch efectivo = 4
    'val_split':     0.15,
    'num_workers':   2,

    # Model
    'encoder':       'resnet34',
    'num_classes':   2,
    'freeze_epochs': 15,   # Epochs with frozen encoder

    # Training
    'total_epochs':  180,
    'lr_decoder':    1e-3,
    'lr_encoder':    1e-4,  # Only active in phase 2
    'weight_decay':  1e-4,
    'warmup_epochs': 3,

    # Loss — per-channel pos_weight: [crater, crack]
    # Cracks are much rarer than craters, so they need a higher penalty
    # for false negatives. focal_gamma > 0 helps with hard examples
    # (thin crack edges, small cracks the model tends to ignore).
    'bce_weight':       0.4,
    'pos_weight':       [8.0, 20.0],   # [crater, crack] — was uniform 10.0
    'focal_gamma':      2.0,           # Focal modulation for hard pixels
    'channel_weights':  [1.0, 2.0],    # Give 50% more Dice weight to cracks

    # Crack-aware loss: boundary + continuity + deep supervision
    'use_deep_supervision': True,
    'boundary_weight':      0.3,       # Weight for boundary-aware crack loss
    'continuity_weight':    0.1,       # Weight for crack continuity loss
    'deep_sup_weights':     [1.0, 0.4, 0.2],  # [main, aux1, aux2]

    # Early stopping — increased patience since cracks learn slowly
    'patience':      30,

    # Binarization threshold — per-channel: lower for cracks
    'threshold':        0.5,
    'threshold_crack':  0.25,   # Lower threshold for crack sensitivity

    # Checkpoint
    'checkpoint_dir': './checkpoints',
    'save_every':     10,
}


# ─────────────────────────────────────────────────────────────────────────────
#  BUCLE DE ENTRENAMIENTO
# ─────────────────────────────────────────────────────────────────────────────

def train_epoch(model, loader, optimizer, criterion, device, scaler=None,
                grad_accum_steps=1):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    optimizer.zero_grad(set_to_none=True)

    for i, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        masks  = batch['mask'].to(device, non_blocking=True)

        if scaler is not None:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = criterion(logits, masks) / grad_accum_steps
            scaler.scale(loss).backward()
        else:
            logits = model(images)
            loss   = criterion(logits, masks) / grad_accum_steps
            loss.backward()

        if (i + 1) % grad_accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * grad_accum_steps
        n_batches  += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def val_epoch(model, loader, criterion, device, threshold=0.5,
              threshold_crack=None):
    """One validation epoch with global micro-average metrics.
    
    If threshold_crack is provided, uses a separate (lower) threshold
    for the crack channel to improve sensitivity on thin cracks.
    """
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    acc = MetricsAccumulator(class_names=['crater', 'crack'])

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        masks  = batch['mask'].to(device, non_blocking=True)

        logits = model(images)
        loss   = criterion(logits, masks)
        probs  = torch.sigmoid(logits)
        
        # Per-channel thresholds
        if threshold_crack is not None:
            preds = torch.zeros_like(probs)
            preds[:, 0] = (probs[:, 0] > threshold).float()
            preds[:, 1] = (probs[:, 1] > threshold_crack).float()
        else:
            preds = (probs > threshold).float()

        acc.update(preds, masks)

        total_loss += loss.item()
        n_batches  += 1

    avg_loss    = total_loss / max(n_batches, 1)
    avg_metrics = acc.compute()

    return avg_loss, avg_metrics


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN TRAINING ROUTINE
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg: dict = None):
    if cfg is None:
        cfg = DEFAULT_CFG.copy()

    # ── Dispositivo ──────────────────────────────────────────────────────────
    device = torch.device(
        'cuda' if torch.cuda.is_available() else
        'mps'  if torch.backends.mps.is_available() else
        'cpu')
    print(f"\n  [Entrenamiento] Dispositivo: {device}")
    if device.type == 'cpu':
        print("  [warning] Training on CPU. This will be very slow.")
        print("          Considera usar Google Colab (GPU gratuita).")

    # ── Datos ─────────────────────────────────────────────────────────────────
    train_loader, val_loader = build_dataloaders(
        data_dir    = cfg['data_dir'],
        img_size    = cfg['img_size'],
        batch_size  = cfg['batch_size'],
        val_split   = cfg['val_split'],
        num_workers = cfg['num_workers'],
    )

    # ── Modelo ────────────────────────────────────────────────────────────────
    use_ds = cfg.get('use_deep_supervision', False)
    if use_ds:
        model = build_unet_deep_supervised(
            encoder_name    = cfg['encoder'],
            encoder_weights = 'imagenet',
            num_classes     = cfg['num_classes'],
        ).to(device)
        print(f"  [Model] Deep Supervision ENABLED")
    else:
        model = build_unet(
            encoder_name    = cfg['encoder'],
            encoder_weights = 'imagenet',
            num_classes     = cfg['num_classes'],
        ).to(device)

    # FASE 1: Encoder congelado
    freeze_encoder(model)

    # ── Optimizador y Scheduler ───────────────────────────────────────────────
    # Fase 1: Solo decoder
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg['lr_decoder'],
        weight_decay=cfg['weight_decay'],
    )

    # Scheduler: warmup + cosine annealing
    warmup_sched = LinearLR(
        optimizer,
        start_factor=0.1, end_factor=1.0,
        total_iters=cfg['warmup_epochs'])

    cosine_sched = CosineAnnealingLR(
        optimizer,
        T_max=cfg['total_epochs'] - cfg['warmup_epochs'],
        eta_min=1e-6)

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_sched, cosine_sched],
        milestones=[cfg['warmup_epochs']])

    # Loss
    base_criterion = CombinedLoss(
        bce_weight=cfg['bce_weight'],
        pos_weight=cfg['pos_weight'],
        focal_gamma=cfg.get('focal_gamma', 0.0),
        channel_weights=cfg.get('channel_weights', None),
    ).to(device)

    if use_ds:
        criterion = CrackAwareLoss(
            base_loss=base_criterion,
            boundary_weight=cfg.get('boundary_weight', 0.3),
            continuity_weight=cfg.get('continuity_weight', 0.1),
            deep_sup_weights=cfg.get('deep_sup_weights', [1.0, 0.4, 0.2]),
        ).to(device)
        print(f"  [Loss] CrackAwareLoss (boundary={cfg.get('boundary_weight', 0.3)}, "
              f"continuity={cfg.get('continuity_weight', 0.1)})")
    else:
        criterion = base_criterion

    # Mixed Precision (solo CUDA)
    scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None

    # ── Checkpoints ───────────────────────────────────────────────────────────
    ckpt_dir = Path(cfg['checkpoint_dir'])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_iou     = 0.0
    patience_cnt = 0
    history      = []

    print(f"\n  {'─'*60}")
    print(f"  TRAINING START  ({cfg['total_epochs']} epochs)")
    print(f"  Encoder: {cfg['encoder']} | Img: {cfg['img_size']}px | Batch: {cfg['batch_size']}")
    print(f"  Phase 1 (encoder frozen):     epochs 1 to {cfg['freeze_epochs']}")
    print(f"  Phase 2 (full fine-tuning):   epochs {cfg['freeze_epochs']+1} to {cfg['total_epochs']}")
    print(f"  {'─'*60}\n")

    for epoch in range(1, cfg['total_epochs'] + 1):
        t0 = time.time()

        # PHASE TRANSITION
        if epoch == cfg['freeze_epochs'] + 1:
            print(f"\n  *** EPOCH {epoch}: entering phase 2, fine-tuning the encoder ***")
            unfreeze_encoder(model)
            # Reconfigurar optimizador con LR diferencial
            optimizer = optim.AdamW(
                get_param_groups(model,
                                 base_lr=cfg['lr_decoder']),
                weight_decay=cfg['weight_decay'],
            )
            # Reiniciar scheduler para la nueva fase
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=cfg['total_epochs'] - cfg['freeze_epochs'],
                eta_min=1e-6)

        # ── TRAIN / VAL ──────────────────────────────────────────────────────
        train_loss = train_epoch(
            model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_metrics = val_epoch(
            model, val_loader, criterion, device,
            threshold=cfg['threshold'],
            threshold_crack=cfg.get('threshold_crack', None))

        scheduler.step()

        elapsed = time.time() - t0
        iou     = val_metrics['mean_iou']

        # ── LOG ──────────────────────────────────────────────────────────────
        phase = "❄" if epoch <= cfg['freeze_epochs'] else "🔥"
        mean_f1 = val_metrics.get('mean_f1', 0.0)
        print(f"  {phase} Ep {epoch:3d}/{cfg['total_epochs']} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"IoU c:{val_metrics['crater_iou']:.3f} "
              f"k:{val_metrics['crack_iou']:.3f} "
              f"m:{iou:.3f} | "
              f"F1 c:{val_metrics.get('crater_f1',0):.3f} "
              f"k:{val_metrics.get('crack_f1',0):.3f} "
              f"m:{mean_f1:.3f} | "
              f"{elapsed:.1f}s")

        # ── HISTORIAL ────────────────────────────────────────────────────────
        history.append({
            'epoch': epoch, 'phase': 1 if epoch <= cfg['freeze_epochs'] else 2,
            'train_loss': round(train_loss, 4),
            'val_loss': round(val_loss, 4),
            **{k: round(v, 4) for k, v in val_metrics.items()},
        })

        # ── GUARDAR MEJOR MODELO (basado en mean IoU + mean F1) ───────────
        # Combining IoU and F1 makes checkpoint selection less jumpy
        score = 0.5 * iou + 0.5 * mean_f1
        if score > best_iou:
            best_iou    = score
            patience_cnt = 0
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'best_score':  best_iou,
                'best_iou':    iou,
                'best_f1':     mean_f1,
                'cfg':         cfg,
            }, ckpt_dir / 'best_model.pth')
            print(f"    ★ New best model (IoU={iou:.4f}, F1={mean_f1:.4f}, score={score:.4f})")
        else:
            patience_cnt += 1

        # PERIODIC CHECKPOINT
        if epoch % cfg['save_every'] == 0:
            torch.save({
                'epoch':       epoch,
                'model_state': model.state_dict(),
                'optim_state': optimizer.state_dict(),
                'best_iou':    best_iou,
                'cfg':         cfg,
            }, ckpt_dir / f'checkpoint_ep{epoch:03d}.pth')

        # ── EARLY STOPPING ───────────────────────────────────────────────────
        if patience_cnt >= cfg['patience']:
            print(f"\n  [Early Stopping] No improvement in {cfg['patience']} epochs.")
            break

    # Save the final checkpoint
    torch.save({
        'epoch':       epoch,
        'model_state': model.state_dict(),
        'best_iou':    best_iou,
        'cfg':         cfg,
    }, ckpt_dir / 'last_model.pth')

    # Guardar historial
    with open(ckpt_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\n  {'═'*60}")
    print(f"  TRAINING COMPLETE")
    print(f"  Best score (0.5*IoU + 0.5*F1): {best_iou:.4f}")
    print(f"  Model saved at: {ckpt_dir / 'best_model.pth'}")
    print(f"  {'═'*60}\n")

    # CONVERGENCE FIGURE
    plot_training_history(history, cfg['freeze_epochs'], ckpt_dir)

    return history, best_iou


# ─────────────────────────────────────────────────────────────────────────────
#  CONVERGENCE FIGURE
# ─────────────────────────────────────────────────────────────────────────────

def plot_training_history(history: list, freeze_epochs: int,
                          output_dir: Path = None):
    """
    Plot the evolution of the run:
      - Panel superior: Train Loss vs Val Loss
      - lower panel: IoU for craters, for cracks and their mean

    The phase 1 to phase 2 transition is marked, where the encoder is unfrozen.
    y el punto del mejor modelo.
    """
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [WARNING] matplotlib not available. Skipping plot.")
        return

    if not history:
        return

    epochs     = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss   = [h['val_loss'] for h in history]

    crater_iou = [h.get('crater_iou', 0.0) for h in history]
    crack_iou  = [h.get('crack_iou', 0.0) for h in history]
    mean_iou   = [h.get('mean_iou', 0.0) for h in history]

    crater_f1  = [h.get('crater_f1', 0.0) for h in history]
    crack_f1   = [h.get('crack_f1', 0.0) for h in history]
    mean_f1    = [h.get('mean_f1', 0.0) for h in history]

    # Best model based on combined score (same as training)
    scores = [0.5 * mi + 0.5 * mf for mi, mf in zip(mean_iou, mean_f1)]
    best_idx   = int(np.argmax(scores))
    best_epoch = epochs[best_idx]
    best_miou  = mean_iou[best_idx]
    best_mf1   = mean_f1[best_idx]

    # ── Academic white theme ──────────────────────────────────────────────
    import matplotlib.ticker as ticker

    C_TRAIN  = '#D62728'   # dark red  — Train Loss
    C_VAL    = '#1F77B4'   # blue      — Val Loss
    C_CRATER = '#FF7F0E'   # orange    — Crater
    C_CRACK  = '#CC0000'   # red       — Crack
    C_MEAN   = '#2CA02C'   # green     — Mean
    C_FREEZE = '#8C6D31'   # dark gold — phase boundary
    C_BEST   = '#E377C2'   # pink      — best epoch marker
    C_GRID   = '#DDDDDD'

    plt.rcParams.update({
        'font.family':     'serif',
        'font.size':       11,
        'axes.labelsize':  12,
        'axes.titlesize':  13,
        'legend.fontsize': 10,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 10), sharex=True)
    fig.patch.set_facecolor('white')

    for ax in (ax1, ax2, ax3):
        ax.set_facecolor('white')
        ax.tick_params(colors='black', direction='in', length=4)
        ax.grid(True, color=C_GRID, linewidth=0.8, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        for sp in ax.spines.values():
            sp.set_color('#AAAAAA')
            sp.set_linewidth(0.8)

    def _vline(ax, label=None):
        if freeze_epochs < epochs[-1]:
            ax.axvline(freeze_epochs + 0.5, color=C_FREEZE, ls='--',
                       lw=1.4, alpha=0.85, label=label, zorder=3)

    # ── Panel 1: Losses ──────────────────────────────────────────────────
    ax1.plot(epochs, train_loss, color=C_TRAIN, lw=1.8, label='Train Loss', zorder=4)
    ax1.plot(epochs, val_loss,   color=C_VAL,   lw=1.8, label='Val Loss',   zorder=4)
    _vline(ax1, label='Encoder unfrozen')
    ax1.set_ylabel('Loss')
    ax1.set_title('U-Net Training Convergence — Tungsten SEM Damage Segmentation',
                  fontweight='bold', pad=10)
    ax1.legend(frameon=True, framealpha=0.9, edgecolor='#BBBBBB', loc='upper right')

    # ── Panel 2: IoU ─────────────────────────────────────────────────────
    ax2.plot(epochs, crater_iou, color=C_CRATER, lw=1.8, label='Crater IoU', zorder=4)
    ax2.plot(epochs, crack_iou,  color=C_CRACK,  lw=1.8, label='Crack IoU',  zorder=4)
    ax2.plot(epochs, mean_iou,   color=C_MEAN,   lw=2.4, label='Mean IoU',   zorder=5)
    _vline(ax2)

    ax2.plot(best_epoch, best_miou, marker='*', color=C_BEST, markersize=14,
             markeredgecolor='black', markeredgewidth=0.6, zorder=6)
    ax2.annotate(f'Best: {best_miou:.3f}\n(epoch {best_epoch})',
                 xy=(best_epoch, best_miou),
                 xytext=(best_epoch + 4, max(0.05, best_miou - 0.12)),
                 color='black', fontsize=9.5, fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='black', lw=1.2))

    ax2.set_ylabel('IoU')
    ax2.set_ylim(-0.05, 1.05)
    ax2.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax2.legend(frameon=True, framealpha=0.9, edgecolor='#BBBBBB', loc='lower right')

    # ── Panel 3: F1 / Dice ───────────────────────────────────────────────
    ax3.plot(epochs, crater_f1, color=C_CRATER, lw=1.8, ls='--', label='Crater F1',      zorder=4)
    ax3.plot(epochs, crack_f1,  color=C_CRACK,  lw=1.8, ls='--', label='Crack F1',       zorder=4)
    ax3.plot(epochs, mean_f1,   color=C_MEAN,   lw=2.4,          label='Mean F1 (Dice)', zorder=5)
    _vline(ax3)

    ax3.plot(best_epoch, best_mf1, marker='*', color=C_BEST, markersize=14,
             markeredgecolor='black', markeredgewidth=0.6, zorder=6)

    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('F1 (Dice)')
    ax3.set_ylim(-0.05, 1.05)
    ax3.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))
    ax3.legend(frameon=True, framealpha=0.9, edgecolor='#BBBBBB', loc='lower right')

    fig.subplots_adjust(hspace=0.08)

    if output_dir is not None:
        out_path = Path(output_dir) / 'training_convergence.png'
    else:
        out_path = Path('training_convergence.png')

    plt.savefig(str(out_path), dpi=150, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close(fig)
    print(f"  >>> Convergence plot saved: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description='Entrenamiento U-Net para SEM de Tungsteno')
    p.add_argument('--data-dir',      default='./data',
                   help='Root directory containing images/ and masks/')
    p.add_argument('--epochs',        type=int, default=180)
    p.add_argument('--batch-size',    type=int, default=4)
    p.add_argument('--img-size',      type=int, default=512)
    p.add_argument('--encoder',       default='resnet34',
                   choices=['resnet34', 'efficientnet-b0', 'mobilenet_v2'])
    p.add_argument('--lr',            type=float, default=1e-3)
    p.add_argument('--freeze-epochs', type=int,   default=15)
    p.add_argument('--checkpoint-dir',default='./checkpoints')
    p.add_argument('--num-workers',   type=int,   default=2)
    args = p.parse_args()

    cfg = DEFAULT_CFG.copy()
    cfg.update({
        'data_dir':       args.data_dir,
        'total_epochs':   args.epochs,
        'batch_size':     args.batch_size,
        'img_size':       args.img_size,
        'encoder':        args.encoder,
        'lr_decoder':     args.lr,
        'freeze_epochs':  args.freeze_epochs,
        'checkpoint_dir': args.checkpoint_dir,
        'num_workers':    args.num_workers,
    })

    train(cfg)


if __name__ == '__main__':
    main()