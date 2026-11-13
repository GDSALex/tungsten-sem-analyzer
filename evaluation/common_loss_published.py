"""Common loss of the submitted training run (reviewer comment 9, revised Figure 3).

The submitted Figure 3 compared the running training loss, averaged over augmented
batches while the weights were changing, with the validation loss at the end of each
epoch.  Both contain the same terms for the submitted model (it has no auxiliary heads),
but they are not measured in the same way.  Here the saved checkpoints of that run
(every 10 epochs from 40 to 150, the selected epoch 125 and the last epoch 155) are
evaluated with one loss, the CombinedLoss of the model, in evaluation mode and without
augmentation, on the 85 training images and on the 14 images held out from training.

Output: results/common_loss_published.csv
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import rev_common as RC  # noqa: E402

import pandas as pd
import torch
from torch.utils.data import DataLoader

from tungsten_analyzer.dataset import TungstenSEMDataset, get_transforms, _FilteredDataset
from tungsten_analyzer.model import CombinedLoss, build_unet

LOSS = dict(bce_weight=0.4, pos_weight=[8.0, 20.0], focal_gamma=2.0, channel_weights=[1.0, 2.0])


def loader(names):
    base = TungstenSEMDataset(image_dir=str(RC.IMG_DIR), transform=get_transforms("val", 512),
                              strip_fraction=RC.STRIP_FRACTION)
    ds = _FilteredDataset(base, allowed_names={f"{n}.jpg" for n in names})
    assert len(ds) == len(names)
    return DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)


@torch.no_grad()
def mean_loss(model, dl, crit, device):
    tot, n = 0.0, 0
    for b in dl:
        x, y = b["image"].to(device), b["mask"].to(device)
        tot += crit(model(x), y).item() * x.shape[0]
        n += x.shape[0]
    return tot / n


def main():
    RC.set_low_priority_off()
    man = RC.load_manifest()
    tr = sorted(man.loc[man.orig_split == "train", "name"])
    va = sorted(man.loc[man.orig_split == "validation", "name"])
    assert len(tr) == 85 and len(va) == 14
    device = torch.device("cuda")
    crit = CombinedLoss(**LOSS).to(device)
    dl_tr, dl_va = loader(tr), loader(va)
    hist = {e["epoch"]: e for e in json.load(open(RC.CKPT_DIR / "training_history.json"))}
    files = [RC.CKPT_DIR / f"checkpoint_ep{e:03d}.pth" for e in range(40, 151, 10)]
    files += [RC.CKPT_DIR / "best_model.pth", RC.CKPT_DIR / "last_model.pth"]
    rows = []
    model = build_unet("resnet34", None, 2).to(device)
    for f in files:
        ck = torch.load(f, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        model.eval()
        ep = int(ck["epoch"])
        r = {"epoch": ep, "file": f.name,
             "train_common_noaug": mean_loss(model, dl_tr, crit, device),
             "val_common": mean_loss(model, dl_va, crit, device),
             "history_train_loss": hist[ep]["train_loss"], "history_val_loss": hist[ep]["val_loss"]}
        rows.append(r)
        print(r, flush=True)
    df = pd.DataFrame(rows).sort_values("epoch")
    df["gap"] = df["val_common"] - df["train_common_noaug"]
    df.to_csv(RC.RESULTS / "common_loss_published.csv", index=False)
    print(df.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
