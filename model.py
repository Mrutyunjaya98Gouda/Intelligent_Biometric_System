"""
FusionModel — Multimodal Biometric Fusion with EfficientNet + ViT
=================================================================
Two EfficientNet backbones extract features from fingerprint and vein images.
Their outputs are projected into a token sequence and fused through a
Transformer encoder with cross-modal attention.
The final output is a 512-dimensional L2-normalised identity embedding.

Fixes / optimisations applied
------------------------------
* Backbone freezing       : EfficientNet weights are frozen for the first
                            `freeze_epochs` training epochs, then unfrozen
                            via unfreeze_backbones().  This prevents
                            catastrophic forgetting on small datasets (24 samples).
* Asymmetric backbones    : Fingerprint uses EfficientNet-B0 (1280-d);
                            vein uses EfficientNet-B1 (1280-d, identical output dim
                            but different capacity) since vein images have different
                            texture statistics.  Swap both to B0 if VRAM is tight.
* Learned modal weighting : Replaced hard mean-pool with a learned 2-weight
                            softmax so the model can discover which modality is
                            more discriminative per sample.
* Dropout scaled to data  : dropout set to 0.05 (was 0.1) — with only 24 samples
                            aggressive dropout prevents the Transformer from
                            learning any structure.
* Gradient checkpointing  : call model.enable_gradient_checkpointing() to trade
                            compute for VRAM when batch size or image resolution
                            is increased.
* CLS token               : A learnable [CLS] token is prepended so the
                            Transformer has a dedicated aggregation position,
                            matching standard ViT practice and giving a richer
                            summary than mean-pooling two tokens.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FusionModel(nn.Module):
    """
    Multimodal fusion model for biometric identification.

    Architecture
    ------------
    1. Two EfficientNet backbones (pretrained, classification heads removed).
       Fingerprint → B0 (1280-d),  Vein → B1 (1280-d).
    2. Linear projection: 1280-d → ``embed_dim`` (default 256) per modality.
    3. Learnable [CLS] token prepended → 3-token sequence.
    4. ViT-style Transformer Encoder (cross-modal attention).
    5. [CLS] output → Linear → L2-normalised ``output_dim``-d embedding.

    Parameters
    ----------
    embed_dim     : int   – internal token dimension (default 256).
    num_heads     : int   – attention heads in the Transformer (default 8).
    num_layers    : int   – Transformer encoder layers (default 2).
    output_dim    : int   – final embedding dimension (default 512).
    dropout       : float – Transformer dropout (default 0.05).
    freeze_epochs : int   – epochs to keep backbones frozen (default 5).
                            Call unfreeze_backbones() after this many epochs.
    """

    def __init__(
        self,
        embed_dim:     int   = 256,
        num_heads:     int   = 8,
        num_layers:    int   = 2,
        output_dim:    int   = 512,
        dropout:       float = 0.05,
        freeze_epochs: int   = 5,
    ):
        super().__init__()
        self.freeze_epochs = freeze_epochs
        backbone_dim = 1280  # both B0 and B1 output 1280-d with num_classes=0

        # ── Backbones ────────────────────────────────────────────────────
        # B0 for fingerprint (sharper ridge detail)
        self.fingerprint_backbone = timm.create_model(
            "efficientnet_b0", pretrained=True, num_classes=0
        )
        # B1 for vein (richer capacity for low-contrast patterns)
        # Change to "efficientnet_b0" if VRAM is limited
        self.vein_backbone = timm.create_model(
            "efficientnet_b1", pretrained=True, num_classes=0
        )

        # Freeze backbones initially to protect pretrained features
        self._set_backbones_frozen(True)

        # ── Projection heads ─────────────────────────────────────────────
        self.fp_proj = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )
        self.vein_proj = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
        )

        # ── CLS token + positional embeddings (3 positions: CLS, FP, Vein) ──
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Parameter(torch.randn(1, 3, embed_dim) * 0.02)

        # ── Transformer Encoder ──────────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=1024,
            dropout=dropout,          # lowered from 0.1 → 0.05
            activation="gelu",
            batch_first=True,
            norm_first=True,          # Pre-LN: more stable for small datasets
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ── Learned modal weighting (applied before Transformer) ─────────
        # A 2-weight softmax over [fp_token, vein_token] lets the model
        # discover which modality is more reliable per forward pass.
        self.modal_weight = nn.Parameter(torch.ones(2))   # [w_fp, w_vein]

        # ── Final embedding head ─────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, output_dim),
        )

        # Gradient checkpointing flag
        self._grad_ckpt = False

    # ------------------------------------------------------------------
    # Backbone freeze / unfreeze helpers
    # ------------------------------------------------------------------

    def _set_backbones_frozen(self, frozen: bool) -> None:
        for backbone in (self.fingerprint_backbone, self.vein_backbone):
            for param in backbone.parameters():
                param.requires_grad = not frozen

    def unfreeze_backbones(self) -> None:
        """Call from train loop after ``freeze_epochs`` to enable fine-tuning."""
        self._set_backbones_frozen(False)
        print("[FusionModel] Backbones unfrozen — full fine-tuning enabled.")

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def enable_gradient_checkpointing(self) -> None:
        """Trade compute for VRAM.  Call before training if hitting OOM."""
        self._grad_ckpt = True
        # timm models expose this natively
        if hasattr(self.fingerprint_backbone, "set_grad_checkpointing"):
            self.fingerprint_backbone.set_grad_checkpointing(True)
        if hasattr(self.vein_backbone, "set_grad_checkpointing"):
            self.vein_backbone.set_grad_checkpointing(True)
        print("[FusionModel] Gradient checkpointing enabled.")

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, fingerprint: torch.Tensor, vein: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        fingerprint : (B, 3, 224, 224)
        vein        : (B, 3, 224, 224)

        Returns
        -------
        embedding : (B, output_dim) – L2-normalised identity embedding
        """
        B = fingerprint.size(0)

        # ── Extract backbone features  →  (B, 1280) ──────────────────────
        fp_feat = self.fingerprint_backbone(fingerprint)
        vn_feat = self.vein_backbone(vein)

        # ── Project  →  (B, embed_dim) ───────────────────────────────────
        fp_tok  = self.fp_proj(fp_feat)
        vn_tok  = self.vein_proj(vn_feat)

        # ── Learned modal weighting ───────────────────────────────────────
        weights = F.softmax(self.modal_weight, dim=0)   # [w_fp, w_vein]
        fp_tok  = fp_tok  * weights[0]
        vn_tok  = vn_tok  * weights[1]

        # ── Prepend CLS token  →  (B, 3, embed_dim) ──────────────────────
        cls = self.cls_token.expand(B, -1, -1)          # (B, 1, embed_dim)
        tokens = torch.cat([cls, fp_tok.unsqueeze(1), vn_tok.unsqueeze(1)], dim=1)
        tokens = tokens + self.pos_embed                # (B, 3, embed_dim)

        # ── Transformer (cross-modal attention)  →  (B, 3, embed_dim) ────
        fused = self.transformer(tokens)

        # ── Use CLS output as the summary representation ──────────────────
        cls_out = fused[:, 0, :]                        # (B, embed_dim)

        # ── Final projection  →  (B, output_dim) ─────────────────────────
        embedding = self.head(cls_out)

        # ── L2 normalise ─────────────────────────────────────────────────
        return F.normalize(embedding, p=2, dim=1)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def count_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    model = FusionModel()
    dummy_fp = torch.randn(2, 3, 224, 224)
    dummy_vn = torch.randn(2, 3, 224, 224)

    out = model(dummy_fp, dummy_vn)
    print(f"Output shape      : {out.shape}")            # (2, 512)
    print(f"L2 norm           : {out.norm(dim=1)}")      # ≈ [1.0, 1.0]
    print(f"Trainable params  : {model.count_trainable_params():,}")
    print(f"Total params      : {model.count_total_params():,}")

    # Unfreeze and recount
    model.unfreeze_backbones()
    print(f"Trainable (unfrozen): {model.count_trainable_params():,}")