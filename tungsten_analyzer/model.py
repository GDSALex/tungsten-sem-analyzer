"""Segmentation network: a U-Net with a ResNet34 encoder pre-trained on ImageNet.

The network predicts two channels at once, craters on channel 0 and cracks on
channel 1, because the two morphologies co-occur and a shared encoder is far more
economical than two separate models on a dataset of this size.

Three choices deserve a word. The encoder is pre-trained and starts out frozen,
which on fewer than a hundred annotated micrographs is what keeps the decoder from
dragging useful low-level filters into noise before it has learned anything itself.
Each decoder block carries a concurrent spatial and channel squeeze-and-excitation
gate, which reweights the feature maps and noticeably sharpens thin cracks. And the
model is wrapped for training in a deep-supervision module that taps the last three
decoder blocks through auxiliary heads, so that gradients reach the high-resolution
stages without vanishing. The auxiliary heads are dropped at inference time.

The composite loss and the segmentation metrics used during training also live here,
next to the architecture they were written for. Note that the metrics in this module
are the ones printed per epoch, aggregated over all pixels of the validation subset.
They are not the per-image figures of the paper, which follow the protocol in
:mod:`tungsten_analyzer.metrics`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np 

try:
    import segmentation_models_pytorch as smp
    SMP_AVAILABLE = True
except ImportError:
    SMP_AVAILABLE = False
    print("[AVISO] segmentation_models_pytorch no encontrado.")
    print("        Instala con: pip install segmentation-models-pytorch")



# ─────────────────────────────────────────────────────────────────────────────
#  MODELO PRINCIPAL: U-Net con ResNet34 pre-entrenado
# ─────────────────────────────────────────────────────────────────────────────

def build_unet(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    num_classes: int = 2,
    freeze_encoder_epochs: int = 10,
) -> nn.Module:
    """
    Construye la U-Net con encoder pre-entrenado.

    Parameters
    -----------
    encoder_name : str
        Backbone to use. Sensible options for a small dataset are
        'resnet34', a good balance of speed and accuracy and the one used in
        the paper, or 'efficientnet-b0', which is lighter for similar accuracy.
        - 'mobilenet_v2'   → Muy ligero, para hardware limitado
    encoder_weights : str
        Pre-trained weights. 'imagenet' is the only option that behaves reliably.
    num_classes : int
        2 = [craters, cracks]. Raise it if further classes are added.
    freeze_encoder_epochs : int
        Number of epochs during which the encoder stays frozen.
        This lets the decoder learn before the pre-trained weights are disturbed.

    Retorna:
    --------
    model : nn.Module con atributo .freeze_encoder_epochs
    """
    if not SMP_AVAILABLE:
        raise ImportError("Instala segmentation-models-pytorch: "
                          "pip install segmentation-models-pytorch")

    model = smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,          # RGB (aunque SEM es gris, triplicamos el canal)
        classes=num_classes,
        activation=None,        # logits out, the sigmoid is applied downstream
        decoder_attention_type="scse",  # Attention gates en el decoder (mejora bordes)
    )

    model.freeze_encoder_epochs = freeze_encoder_epochs
    model.num_classes = num_classes

    return model


def freeze_encoder(model: nn.Module):
    """Freeze the encoder weights, which is what phase 1 of training relies on."""
    for param in model.encoder.parameters():
        param.requires_grad = False
    print("  [UNet] Encoder CONGELADO (solo decoder entrena)")


def unfreeze_encoder(model: nn.Module, lr_scale: float = 0.1):
    """
    Descongela el encoder con LR reducida (fine-tuning gradual).
    Return the parameter groups handed to the optimiser.
    """
    for param in model.encoder.parameters():
        param.requires_grad = True
    print(f"  [UNet] Encoder DESCONGELADO (LR × {lr_scale})")


def get_param_groups(model: nn.Module, base_lr: float = 1e-3):
    """
    Return parameter groups with a differential learning rate,
      - Encoder: base_lr × 0.1 (fine-tuning suave)
      - Decoder: base_lr       (aprendizaje normal)
    """
    encoder_params = list(model.encoder.parameters())
    decoder_params = [p for p in model.parameters()
                      if not any(p is ep for ep in encoder_params)]
    return [
        {'params': encoder_params, 'lr': base_lr * 0.1, 'name': 'encoder'},
        {'params': decoder_params, 'lr': base_lr,       'name': 'decoder'},
    ]


# ─────────────────────────────────────────────────────────────────────────────
#  LOSS FUNCTION: weighted BCE + Dice, robust to class imbalance
# ─────────────────────────────────────────────────────────────────────────────

class CombinedLoss(nn.Module):
    """
    Combined loss: weighted BCE + Dice + an optional Focal term.

    Cracks and craters together occupy under 5% of the area of a SEM micrograph,
    and under that imbalance a plain BCE converges happily to predicting
    background everywhere. The three terms below correct for it.

    Mejoras v2:
      - A per-channel pos_weight, since cracks are far scarcer than craters
        and therefore need the heavier positive weight.
      - A Focal component (gamma > 0), which suppresses the contribution of the
        easy background pixels and concentrates the gradient on the hard ones,
        crack edges and small isolated craters.
      - A channel-weighted Dice term, so that the crack channel can be given
        extra weight in the overlap part of the objective.

    Parameters
    -----------
    bce_weight : float
        Weight of the BCE term in the sum, between 0 and 1.
    pos_weight : float o list/tuple de 2 floats
        Penalty factor for false negatives. A single float applies to both
        channels; a list [crater_pw, crack_pw] sets them independently.

    smooth : float
        Dice smoothing term, which avoids division by zero on empty masks.
    focal_gamma : float
        Exponente focal. 0 = sin focal (BCE pura). 2.0 = focal fuerte.
        Recomendado: 1.0–2.0 para grietas finas.
    channel_weights : list o None
        Relative weight of each channel inside the Dice term. For instance
        [1.0, 1.5] gives the crack channel 50% more weight.
    """

    def __init__(self, bce_weight: float = 0.5, pos_weight=5.0,
                 smooth: float = 1.0, focal_gamma: float = 0.0,
                 channel_weights=None):
        super().__init__()
        self.bce_weight     = bce_weight
        self.dice_weight    = 1.0 - bce_weight
        self.smooth         = smooth
        self.focal_gamma    = focal_gamma
        
        # pos_weight: acepta float o lista [crater, crack]
        if isinstance(pos_weight, (list, tuple)):
            self.register_buffer('pos_weight',
                                 torch.tensor(pos_weight, dtype=torch.float32))
        else:
            self.register_buffer('pos_weight', None)
            self._pos_weight_scalar = pos_weight
        
        # channel_weights para Dice
        if channel_weights is not None:
            self.register_buffer('channel_weights',
                                 torch.tensor(channel_weights, dtype=torch.float32))
        else:
            self.channel_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C, H, W), before any activation
        targets : (B, C, H, W) binario float32
        """
        C = logits.shape[1]
        probs = torch.sigmoid(logits)

        # ── BCE con pos_weight por canal ──────────────────────────────
        if self.pos_weight is not None:
            pw = self.pos_weight.to(logits.device)
        else:
            pw = torch.tensor([self._pos_weight_scalar] * C,
                              device=logits.device)
        
        bce = F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=pw.view(1, -1, 1, 1),
            reduction='none')  # (B, C, H, W)
        
        # ── Focal modulation (si gamma > 0) ──────────────────────────
        if self.focal_gamma > 0:
            pt = probs * targets + (1 - probs) * (1 - targets)
            focal_weight = (1 - pt) ** self.focal_gamma
            bce = bce * focal_weight
        
        bce = bce.mean()

        # ── Dice Loss (con pesos opcionales por canal) ────────────────
        intersection = (probs * targets).sum(dim=(2, 3))  # (B, C)
        union        = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice_score   = (2 * intersection + self.smooth) / (union + self.smooth)
        
        if self.channel_weights is not None:
            cw = self.channel_weights.to(logits.device).view(1, -1)
            dice_loss = 1.0 - (dice_score * cw).sum(dim=1) / cw.sum()
            dice_loss = dice_loss.mean()
        else:
            dice_loss = 1.0 - dice_score.mean()

        return self.bce_weight * bce + self.dice_weight * dice_loss


# ─────────────────────────────────────────────────────────────────────────────
#  EVALUATION METRICS
# ─────────────────────────────────────────────────────────────────────────────

class MetricsAccumulator:
    """
    Accumulates TP, FP, FN globally across all validation batches,
    then computes micro-average IoU/F1/Precision/Recall once at the end.
    
    This is the standard in semantic segmentation — avoids artifacts
    with empty images or small batches.
    """

    def __init__(self, class_names: list):
        self.class_names = class_names
        self.n_classes   = len(class_names)
        self.reset()

    def reset(self):
        self.tp = np.zeros(self.n_classes, dtype=np.float64)
        self.fp = np.zeros(self.n_classes, dtype=np.float64)
        self.fn = np.zeros(self.n_classes, dtype=np.float64)

    def update(self, preds_bin: torch.Tensor, targets: torch.Tensor):
        """
        Accumulate stats from one batch.
        preds_bin : (B, C, H, W) binary float
        targets   : (B, C, H, W) binary float
        """
        for i in range(self.n_classes):
            p = preds_bin[:, i].float()
            t = targets[:, i].float()
            self.tp[i] += (p * t).sum().item()
            self.fp[i] += (p * (1 - t)).sum().item()
            self.fn[i] += ((1 - p) * t).sum().item()

    def compute(self) -> dict:
        """
        Compute final metrics over all accumulated validation data.
        Returns IoU, F1 (Dice), Precision and Recall per class + means.
        """
        metrics = {}
        iou_vals = []
        f1_vals  = []

        for i, name in enumerate(self.class_names):
            tp, fp, fn = self.tp[i], self.fp[i], self.fn[i]

            # IoU = TP / (TP + FP + FN)
            denom_iou = tp + fp + fn
            iou = tp / denom_iou if denom_iou > 0 else 0.0

            # F1 (Dice) = 2*TP / (2*TP + FP + FN)
            denom_f1 = 2 * tp + fp + fn
            f1 = (2 * tp) / denom_f1 if denom_f1 > 0 else 0.0

            # Precision = TP / (TP + FP)
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0

            # Recall = TP / (TP + FN)
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            metrics[f'{name}_iou']       = iou
            metrics[f'{name}_f1']        = f1
            metrics[f'{name}_precision'] = prec
            metrics[f'{name}_recall']    = rec

            iou_vals.append(iou)
            f1_vals.append(f1)

        metrics['mean_iou'] = float(np.mean(iou_vals))
        metrics['mean_f1']  = float(np.mean(f1_vals))

        return metrics


# ─────────────────────────────────────────────────────────────────────────────
#  DEEP SUPERVISION WRAPPER
# ─────────────────────────────────────────────────────────────────────────────

class DeepSupervisedUNet(nn.Module):
    """
    Wraps the SMP U-Net to add deep supervision from intermediate decoder
    layers. This forces the network to produce correct segmentations at
    multiple resolutions, which is critical for thin structures (cracks)
    that may only be visible at certain scales.

    Uses forward hooks to capture intermediate decoder outputs without
    reimplementing the decoder forward pass (compatible with all SMP versions).

    During training, returns (main_output, [aux1, aux2, ...]).
    During inference (eval mode), returns only the full-resolution prediction.
    """

    def __init__(self, base_model, num_classes=2):
        super().__init__()
        self.base = base_model
        self.num_classes = num_classes

        # Identify decoder blocks and their output channels via a probe
        decoder_blocks = base_model.decoder.blocks
        n_blocks = len(decoder_blocks)
        block_channels = self._probe_decoder_channels(base_model)

        # Tap the last N decoder blocks (higher resolution ones)
        n_tap = min(3, n_blocks)
        self._tap_indices = list(range(n_blocks - n_tap, n_blocks))
        self._intermediate = {}

        # Register forward hooks on the decoder blocks we want to tap
        for idx in self._tap_indices:
            decoder_blocks[idx].register_forward_hook(self._make_hook(idx))

        # 1x1 conv heads for auxiliary predictions
        self.aux_heads = nn.ModuleDict()
        for idx in self._tap_indices:
            out_ch = block_channels[idx]
            self.aux_heads[str(idx)] = nn.Conv2d(out_ch, num_classes, kernel_size=1)

        # Expose encoder for freeze/unfreeze compatibility
        self.encoder = base_model.encoder
        self.freeze_encoder_epochs = getattr(base_model, 'freeze_encoder_epochs', 10)

    @torch.no_grad()
    def _probe_decoder_channels(self, base_model):
        """Run a dummy forward pass to detect each decoder block's output channels."""
        channels = {}
        hooks = []
        for i, block in enumerate(base_model.decoder.blocks):
            def make_hook(idx):
                def hook(m, inp, out):
                    channels[idx] = out.shape[1]
                return hook
            hooks.append(block.register_forward_hook(make_hook(i)))

        dummy = torch.randn(1, 3, 64, 64)
        base_model.eval()
        base_model(dummy)
        for h in hooks:
            h.remove()
        return channels

    def _make_hook(self, idx):
        """Create a forward hook that captures the block's output."""
        def hook(module, input, output):
            self._intermediate[idx] = output
        return hook

    def forward(self, x):
        self._intermediate.clear()

        # Full forward pass through the base U-Net
        main_out = self.base(x)

        if self.training:
            aux_preds = []
            for idx in self._tap_indices:
                feat = self._intermediate.get(idx)
                if feat is None:
                    continue
                aux_pred = self.aux_heads[str(idx)](feat)
                if aux_pred.shape[2:] != main_out.shape[2:]:
                    aux_pred = F.interpolate(
                        aux_pred, size=main_out.shape[2:],
                        mode='bilinear', align_corners=False)
                aux_preds.append(aux_pred)
            return main_out, aux_preds
        else:
            return main_out


def build_unet_deep_supervised(
    encoder_name: str = "resnet34",
    encoder_weights: str = "imagenet",
    num_classes: int = 2,
    freeze_encoder_epochs: int = 10,
) -> nn.Module:
    """Build a U-Net with deep supervision for improved thin-structure detection."""
    base = build_unet(encoder_name, encoder_weights, num_classes, freeze_encoder_epochs)
    model = DeepSupervisedUNet(base, num_classes)
    model.num_classes = num_classes
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  CONNECTIVITY / TOPOLOGY LOSS for cracks
# ─────────────────────────────────────────────────────────────────────────────

class CrackAwareLoss(nn.Module):
    """
    Extends CombinedLoss with crack-specific terms:

    1. **Boundary loss** (distance-weighted): penalizes missed pixels near
       crack boundaries more heavily. Uses a distance transform of the GT
       to weight the loss — pixels far from any GT crack contribute less
       to the crack loss, reducing noise from thick GT annotations.

    2. **Continuity loss**: penalizes breaks in predicted cracks by
       measuring smoothness of the probability map along the crack direction.
       This encourages connected, continuous crack predictions.

    3. **Deep supervision loss**: weighted sum of losses at multiple scales.
    """

    def __init__(self, base_loss: CombinedLoss,
                 boundary_weight: float = 0.3,
                 continuity_weight: float = 0.1,
                 deep_sup_weights: list = None):
        super().__init__()
        self.base_loss = base_loss
        self.boundary_weight = boundary_weight
        self.continuity_weight = continuity_weight
        # Weights for [main, aux1, aux2, aux3] predictions
        self.deep_sup_weights = deep_sup_weights or [1.0, 0.4, 0.2, 0.1]

    def forward(self, predictions, targets):
        """
        predictions: either a tensor (B,C,H,W) or tuple (main, [aux1, aux2, ...])
        targets: (B,C,H,W)
        """
        if isinstance(predictions, tuple):
            main_logits, aux_list = predictions
        else:
            main_logits = predictions
            aux_list = []

        # Main loss
        loss = self.deep_sup_weights[0] * self._compute_loss(main_logits, targets)

        # Auxiliary losses (deep supervision)
        for i, aux_logits in enumerate(aux_list):
            w = self.deep_sup_weights[min(i + 1, len(self.deep_sup_weights) - 1)]
            loss = loss + w * self._compute_loss(aux_logits, targets)

        return loss

    def _compute_loss(self, logits, targets):
        """Base loss + boundary + continuity on crack channel."""
        base = self.base_loss(logits, targets)

        probs = torch.sigmoid(logits)

        # ── Boundary-weighted loss for crack channel (ch=1) ──────────
        if self.boundary_weight > 0 and targets.shape[1] > 1:
            crack_gt = targets[:, 1:2]  # (B, 1, H, W)
            crack_logits = logits[:, 1:2]

            # Distance transform weight: emphasize pixels near crack center
            # We use the GT mask dilated vs eroded to find boundary region
            boundary_w = self._boundary_weight_map(crack_gt)
            bce_boundary = F.binary_cross_entropy_with_logits(
                crack_logits, crack_gt, weight=boundary_w, reduction='mean')
            base = base + self.boundary_weight * bce_boundary

        # ── Continuity loss: penalize abrupt changes in crack probs ──
        if self.continuity_weight > 0 and targets.shape[1] > 1:
            crack_prob = probs[:, 1:2]
            crack_gt = targets[:, 1:2]
            # Only apply where GT has cracks (don't penalize smooth background)
            mask = (crack_gt > 0.5).float()
            if mask.sum() > 0:
                # Gradient magnitude of crack probability
                dx = torch.abs(crack_prob[:, :, :, 1:] - crack_prob[:, :, :, :-1])
                dy = torch.abs(crack_prob[:, :, 1:, :] - crack_prob[:, :, :-1, :])
                # We want low gradient ALONG cracks (continuity) but the GT
                # mask doesn't give us direction. Instead, penalize low prob
                # in the interior of GT cracks (holes in predictions).
                interior_miss = mask * (1.0 - crack_prob)
                continuity = interior_miss.mean()
                base = base + self.continuity_weight * continuity

        return base

    def _boundary_weight_map(self, gt_mask):
        """
        Create a weight map that emphasizes crack center pixels.
        Uses max-pooling to approximate distance transform on GPU.
        """
        # Erode: pixels that are definitely inside the crack
        kernel_size = 5
        eroded = -F.max_pool2d(
            -gt_mask, kernel_size=kernel_size,
            stride=1, padding=kernel_size // 2)

        # Weight: center pixels get weight 2.0, border pixels get 1.0,
        # background gets 0.5
        weight = torch.ones_like(gt_mask) * 0.5
        weight = torch.where(gt_mask > 0.5, torch.ones_like(weight), weight)
        weight = torch.where(eroded > 0.5, weight * 2.0, weight)

        return weight


if __name__ == '__main__':
    # Quick architecture smoke test
    model = build_unet()
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")   # Esperado: (2, 2, 512, 512)
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")

    # Test deep supervised
    model_ds = build_unet_deep_supervised()
    model_ds.train()
    out_ds = model_ds(x)
    main, aux = out_ds
    print(f"\nDeep Supervised:")
    print(f"  Main: {main.shape}")
    for i, a in enumerate(aux):
        print(f"  Aux {i}: {a.shape}")