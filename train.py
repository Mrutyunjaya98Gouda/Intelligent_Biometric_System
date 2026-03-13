"""
train.py — Intelligent Biometric System — Training Entry Point
==============================================================
Called by demo.py via:  import train; train.train()

Architecture
------------
  • FusionModel         : EfficientNet-B0/B1 + ViT-style Transformer fusion
  • CombinedLoss        : ArcFace(m=0.65) + Triplet + SupCon
                          + VerificationLoss + IntraClassLoss
  • SessionAwareSampler : guarantees cross-session genuine pairs per batch

Training schedule
-----------------
  Epochs  1–FREEZE_EPOCHS : backbone weights frozen; head trains on all losses
  Epoch   FREEZE_EPOCHS+1 : backbones unfrozen at low LR for fine-tuning
  Epochs  1–WARMUP_EPOCHS : ArcFace weight linearly ramped 0 → 1
                            (verification losses active from epoch 1)

Checkpoint
----------
  Saves best_biometric_model.pth (keyed on best TAR @ FAR=1%)
  → this is the path demo.py loads for eval / verify modes.

Outputs
-------
  best_biometric_model.pth   — best checkpoint (by TAR @ FAR=1%)
  loss_curves.csv            — epoch log (loaded by test.py plot 15)
  loss_curves.png            — training curve figure
"""

import os
import csv
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from dataset import BiometricDataset
from model   import FusionModel
from losses  import CombinedLoss
from sampler import SessionAwareSampler


# ══════════════════════════════════════════════════════════════════════════════
#  Configuration  — edit to match your machine
# ══════════════════════════════════════════════════════════════════════════════
ROOT_DIR        = r"D:\Intelligent_Biometric_System\NUPT-FPV-main\image"

# Batch sampling: n_subjects × n_per_class images per step
N_SUBJECTS      = 4     # distinct identities per batch
N_PER_CLASS     = 3     # samples per identity  →  batch ≈ 48

EPOCHS          = 60
LR              = 1e-4      # head / loss-layer learning rate
BACKBONE_LR     = 5e-6      # backbone fine-tune LR (after unfreeze)
FREEZE_EPOCHS   = 12        # keep backbones frozen for first N epochs
WARMUP_EPOCHS   = 5         # linearly ramp ArcFace weight over first N epochs

DEVICE          = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SCRIPT_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "best_biometric_model.pth")
CSV_SAVE_PATH   = os.path.join(SCRIPT_DIR, "loss_curves.csv")
PNG_SAVE_PATH   = os.path.join(SCRIPT_DIR, "loss_curves.png")

# Loss component weights
W_ARC    = 1.0
W_TRI    = 0.20
W_SUPCON = 0.50
W_VERIF  = 0.30
W_INTRA  = 0.10

# Colour palette (consistent with demo.py / test.py)
C_BLUE   = "#378ADD"
C_RED    = "#E24B4A"
C_GREEN  = "#1D9E75"
C_ORANGE = "#F5A623"
C_GREY   = "#888780"
C_PURPLE = "#7F77DD"
# ══════════════════════════════════════════════════════════════════════════════


# ──────────────────────────────────────────────────────────────────────────────
#  Optimizer builder
#  Backbone params get a lower LR to protect pretrained features.
# ──────────────────────────────────────────────────────────────────────────────
def _build_optimizer(model: FusionModel,
                     criterion: CombinedLoss,
                     lr: float,
                     backbone_lr: float) -> torch.optim.Optimizer:
    backbone_ids = set(
        id(p)
        for p in (list(model.fingerprint_backbone.parameters())
                  + list(model.vein_backbone.parameters()))
    )
    head_params     = [p for p in model.parameters()    if id(p) not in backbone_ids]
    backbone_params = [p for p in model.parameters()    if id(p) in backbone_ids]
    loss_params     = list(criterion.parameters())

    return torch.optim.AdamW(
        [
            {"params": head_params + loss_params,
             "lr": lr,          "weight_decay": 1e-4},
            {"params": backbone_params,
             "lr": backbone_lr, "weight_decay": 1e-4},
        ]
    )


# ──────────────────────────────────────────────────────────────────────────────
#  ArcFace warm-up schedule
# ──────────────────────────────────────────────────────────────────────────────
def _arc_weight(epoch: int) -> float:
    """Linearly ramp ArcFace contribution from 0 → W_ARC over WARMUP_EPOCHS."""
    if epoch <= WARMUP_EPOCHS:
        return W_ARC * (epoch / WARMUP_EPOCHS)
    return W_ARC


# ──────────────────────────────────────────────────────────────────────────────
#  TAR @ FAR evaluation  (runs on validation set each epoch)
# ──────────────────────────────────────────────────────────────────────────────
def _tar_at_far(model: FusionModel,
                val_loader: DataLoader,
                device: torch.device,
                far_target: float = 0.01) -> float:
    """Return TAR @ FAR=far_target computed on validation embeddings."""
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for fp, vn, labels in val_loader:
            emb = model(fp.to(device), vn.to(device))
            all_emb.append(emb.cpu())
            all_lbl.append(labels)

    emb = torch.cat(all_emb).numpy()
    lbl = torch.cat(all_lbl).numpy()
    n   = emb.shape[0]

    nrm = np.linalg.norm(emb, axis=1, keepdims=True)
    e   = emb / np.clip(nrm, 1e-8, None)
    sim = e @ e.T

    genuine, impostor = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (genuine if lbl[i] == lbl[j] else impostor).append(sim[i, j])

    genuine  = np.array(genuine)
    impostor = np.array(impostor)
    if len(genuine) == 0 or len(impostor) == 0:
        return 0.0

    for t in np.sort(impostor)[::-1]:
        if np.mean(impostor >= t) <= far_target:
            return float(np.mean(genuine >= t))
    return 0.0


def _mean_genuine_sim(model: FusionModel,
                      val_loader: DataLoader,
                      device: torch.device) -> float:
    """Average cosine similarity of same-class pairs in the validation set."""
    model.eval()
    all_emb, all_lbl = [], []
    with torch.no_grad():
        for fp, vn, labels in val_loader:
            emb = model(fp.to(device), vn.to(device))
            all_emb.append(emb.cpu())
            all_lbl.append(labels)

    emb = torch.cat(all_emb)
    lbl = torch.cat(all_lbl).numpy()
    sim = F.cosine_similarity(emb.unsqueeze(0), emb.unsqueeze(1), dim=2).numpy()
    n   = emb.shape[0]

    genuine = [sim[i, j]
               for i in range(n) for j in range(i + 1, n)
               if lbl[i] == lbl[j]]
    return float(np.mean(genuine)) if genuine else 0.0


# ──────────────────────────────────────────────────────────────────────────────
#  Loss-curve plot
# ──────────────────────────────────────────────────────────────────────────────
def _save_loss_plot(log: list, out_path: str) -> None:
    epochs  = [r[0] for r in log]
    train_t = [r[1] for r in log]
    val_t   = [r[2] for r in log]
    arc_v   = [r[3] for r in log]
    tri_v   = [r[4] for r in log]
    sc_v    = [r[5] for r in log]
    vf_v    = [r[6] for r in log]
    in_v    = [r[7] for r in log]
    tar_v   = [r[8] * 100 for r in log]
    gen_v   = [r[9] for r in log]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Left: total train vs val
    axes[0].plot(epochs, train_t, color=C_BLUE, linewidth=2, label="Train")
    axes[0].plot(epochs, val_t,   color=C_RED,  linewidth=2, label="Val")
    if FREEZE_EPOCHS < len(epochs):
        axes[0].axvline(FREEZE_EPOCHS + 0.5, color=C_GREY,
                        linestyle="--", linewidth=1,
                        label=f"Unfreeze @ ep {FREEZE_EPOCHS + 1}")
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].set_title("Train vs Val Loss", fontweight="bold")
    axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)

    # Middle: loss components
    for vals, label, color in [
        (arc_v, "ArcFace",      C_PURPLE),
        (tri_v, "Triplet",      C_GREEN),
        (sc_v,  "SupCon",       C_BLUE),
        (vf_v,  "Verification", C_RED),
        (in_v,  "Intra",        C_ORANGE),
    ]:
        axes[1].plot(epochs, vals, linewidth=2, label=label, color=color)
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss Components", fontweight="bold")
    axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)

    # Right: TAR@1% and mean genuine similarity
    ax2 = axes[2].twinx()
    axes[2].plot(epochs, tar_v, color=C_RED,  linewidth=2.5,
                 label="TAR @ FAR=1% (%)")
    ax2.plot(epochs,     gen_v, color=C_BLUE, linewidth=2, linestyle="--",
             label="Mean genuine sim")
    axes[2].set_ylabel("TAR @ FAR=1% (%)", color=C_RED)
    ax2.set_ylabel("Mean genuine cosine sim",  color=C_BLUE)
    axes[2].set_title("Verification Metrics vs Epoch", fontweight="bold")
    axes[2].legend(loc="upper left",  fontsize=9)
    ax2.legend(loc="lower right", fontsize=9)
    axes[2].grid(alpha=0.3)

    plt.suptitle("Training Curves — Biometric Fusion Model",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Loss curves → {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Main training function  (called by demo.py)
# ══════════════════════════════════════════════════════════════════════════════
def train() -> None:
    sep = "═" * 64
    print(f"\n{sep}")
    print("  Biometric Fusion Model — Training")
    print(f"{sep}")
    print(f"  Device        : {DEVICE}")
    print(f"  Dataset       : {ROOT_DIR}")
    print(f"  Epochs        : {EPOCHS}")
    print(f"  Freeze epochs : {FREEZE_EPOCHS}")
    print(f"  Warmup epochs : {WARMUP_EPOCHS}")
    print(f"  Save path     : {MODEL_SAVE_PATH}")
    print(f"{sep}\n")

    # ── Dataset ───────────────────────────────────────────────────────────
    dataset     = BiometricDataset(root_dir=ROOT_DIR)
    num_classes = dataset.num_classes
    print(f"  Subjects : {num_classes}   |   Samples : {len(dataset)}\n")

    if num_classes < 2:
        raise ValueError(
            f"Need at least 2 subjects for training, found {num_classes}.\n"
            f"  Check DATASET_ROOT: {ROOT_DIR}"
        )

    # ── Train / Validation split  (80 / 20) ──────────────────────────────
    total      = len(dataset)
    val_size   = max(1, int(total * 0.20))
    train_size = total - val_size
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # ── Session-aware sampler — build a minimal view for the sampler ──────
    # SessionAwareSampler reads .samples from its dataset argument.
    train_indices = sorted(train_ds.indices)

    class _SamplerView:
        """Exposes only the training subset to SessionAwareSampler."""
        def __init__(self, full_dataset, indices):
            self.samples = [
                (full_dataset.samples[i][0],
                 full_dataset.samples[i][1],
                 full_dataset.samples[i][2])
                for i in indices
            ]

    sampler_view = _SamplerView(dataset, train_indices)
    sampler = SessionAwareSampler(
        sampler_view,
        n_subjects  = min(N_SUBJECTS,  num_classes),
        n_per_class = N_PER_CLASS,
    )

    train_loader = DataLoader(
        train_ds,
        batch_sampler = sampler,
        num_workers   = 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=32, shuffle=False, num_workers=0
    )
    print(f"  Train samples : {train_size}   |   Val samples : {val_size}")
    print(f"  Batch size    : ~{N_SUBJECTS * N_PER_CLASS}  "
          f"({N_SUBJECTS} subjects × {N_PER_CLASS} samples)")
    print(f"  Steps/epoch   : ~{len(sampler)}\n")

    # ── Model ─────────────────────────────────────────────────────────────
    model = FusionModel(
        embed_dim     = 256,
        num_heads     = 8,
        num_layers    = 2,
        output_dim    = 512,
        freeze_epochs = FREEZE_EPOCHS,
    ).to(DEVICE)
    print(f"  Trainable params (frozen backbones): "
          f"{model.count_trainable_params():,}\n")

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = CombinedLoss(
        num_classes  = num_classes,
        s            = 32.0,
        m            = 0.65,
        w_tri        = W_TRI,
        w_supcon     = W_SUPCON,
        w_verif      = W_VERIF,
        w_intra      = W_INTRA,
        pos_margin   = 0.85,
        neg_margin   = 0.20,
        temperature  = 0.07,
    ).to(DEVICE)

    optimizer          = _build_optimizer(model, criterion, LR, BACKBONE_LR)
    best_tar           = -1.0
    backbones_unfrozen = False
    epoch_log          = []

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(1, EPOCHS + 1):

        # Unfreeze backbones after FREEZE_EPOCHS
        if epoch > FREEZE_EPOCHS and not backbones_unfrozen:
            model.unfreeze_backbones()
            optimizer = _build_optimizer(model, criterion, LR, BACKBONE_LR)
            print(f"\n  ↳ Backbones unfrozen.  "
                  f"Trainable: {model.count_trainable_params():,}\n")
            backbones_unfrozen = True

        arc_w = _arc_weight(epoch)

        # ── Train one epoch ───────────────────────────────────────────────
        model.train(); criterion.train(); dataset.train()
        sums      = dict(total=0., arc=0., tri=0., supcon=0., verif=0., intra=0.)
        n_batches = 0

        for fp, vn, labels in train_loader:
            fp, vn, labels = fp.to(DEVICE), vn.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            embeddings = model(fp, vn)
            total, arc, tri, supcon, verif, intra = criterion(embeddings, labels)

            # During warm-up, scale down the ArcFace contribution
            if arc_w < 1.0:
                total = (arc_w    * arc
                         + W_TRI    * tri
                         + W_SUPCON * supcon
                         + W_VERIF  * verif
                         + W_INTRA  * intra)

            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            sums["total"]  += total.item()
            sums["arc"]    += arc.item()
            sums["tri"]    += tri.item()
            sums["supcon"] += supcon.item()
            sums["verif"]  += verif.item()
            sums["intra"]  += intra.item()
            n_batches      += 1

        avg = {k: v / max(n_batches, 1) for k, v in sums.items()}

        # ── Validate ──────────────────────────────────────────────────────
        model.eval(); criterion.eval(); dataset.eval()
        val_loss = 0.0; val_n = 0
        with torch.no_grad():
            for fp, vn, labels in val_loader:
                fp, vn, labels = fp.to(DEVICE), vn.to(DEVICE), labels.to(DEVICE)
                emb = model(fp, vn)
                loss, *_ = criterion(emb, labels)
                val_loss += loss.item(); val_n += 1
        avg_val = val_loss / max(val_n, 1)

        tar_1pct = _tar_at_far(model, val_loader, DEVICE, far_target=0.01)
        mean_gen = _mean_genuine_sim(model, val_loader, DEVICE)

        phase = "frozen " if not backbones_unfrozen else "unfrz "
        print(
            f"Ep [{epoch:>3}/{EPOCHS}][{phase}]  "
            f"Loss:{avg['total']:.3f}  "
            f"Arc:{avg['arc']:.3f}  Tri:{avg['tri']:.3f}  "
            f"SC:{avg['supcon']:.3f}  Vf:{avg['verif']:.3f}  In:{avg['intra']:.3f}"
            f"  |  Val:{avg_val:.3f}  "
            f"TAR@1%:{tar_1pct*100:.1f}%  GenSim:{mean_gen:.3f}"
        )

        epoch_log.append((
            epoch,
            avg["total"], avg_val,
            avg["arc"],   avg["tri"],
            avg["supcon"], avg["verif"], avg["intra"],
            tar_1pct, mean_gen,
        ))

        # Checkpoint — primary metric is TAR @ FAR=1%
        if tar_1pct > best_tar:
            best_tar = tar_1pct
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  ✓ Best TAR@1% = {best_tar*100:.2f}%  → {MODEL_SAVE_PATH}")

    # ── Save outputs ───────────────────────────────────────────────────────
    with open(CSV_SAVE_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_loss", "val_loss",
                    "arc_loss", "tri_loss",
                    "supcon_loss", "verif_loss", "intra_loss",
                    "tar_1pct", "mean_genuine_sim"])
        w.writerows(epoch_log)
    print(f"\n  CSV log → {CSV_SAVE_PATH}")

    _save_loss_plot(epoch_log, PNG_SAVE_PATH)

    print(f"\n{'═'*64}")
    print(f"  Training complete!")
    print(f"  Best TAR @ FAR=1% : {best_tar*100:.2f}%")
    print(f"  Model saved       : {MODEL_SAVE_PATH}")
    print(f"{'═'*64}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Allow running directly:  py train.py
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    train()