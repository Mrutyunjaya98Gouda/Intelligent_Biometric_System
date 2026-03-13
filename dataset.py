"""
BiometricDataset — NUPT-FPV Multimodal Fingerprint + Finger-Vein Dataset
========================================================================
Supports two dataset layouts automatically:

Layout A — NUPT-FPV (new):
    <root>/
      Session1/
        Fingerprint/<subject_id>/<images>
        FingerVein/<subject_id>/<images>
      Session2/
        Fingerprint/<subject_id>/<images>
        FingerVein/<subject_id>/<images>
    Identity = subject folder name (e.g. "001").
    Each subject has images across multiple sessions → multiple samples/class.

Layout B — legacy flat layout (old data):
    <root>/
      Fingerprint/**/<user>_<session>_<finger>_<sample>.bmp
      Fingervein/**/<user>_<session>_<finger>_<sample>.bmp
    Identity parsed from filename prefix.
    Single-user fallback: (finger × session) becomes pseudo-class.

The correct layout is detected automatically at runtime.
"""

import os
import glob
import unicodedata
import cv2
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _sanitize_name(name: str) -> str:
    """Strip zero-width / format Unicode characters (category Cf) from name."""
    name = unicodedata.normalize("NFC", name)
    return "".join(ch for ch in name if unicodedata.category(ch) != "Cf")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class BiometricDataset(Dataset):
    """
    PyTorch Dataset for multimodal biometric fusion.

    Auto-detects NUPT-FPV folder layout vs legacy flat layout.

    Parameters
    ----------
    root_dir  : str  – path to dataset root.
    transform : optional custom transform (applied to both modalities).
    """

    def __init__(self, root_dir: str = None, transform=None):
        if root_dir is None:
            root_dir = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "Data"
            )
        self.root_dir = root_dir

        # Auto-detect layout and build paired sample list
        self.samples, self.label_map = self._build_pairs()
        self.num_classes = len(self.label_map)

        # ── Transforms ───────────────────────────────────────────────────
        if transform is not None:
            self.transform     = transform
            self.val_transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(degrees=5),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])
            self.val_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ])

        self.training_mode = True
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    # ------------------------------------------------------------------
    # Layout detection
    # ------------------------------------------------------------------

    def _is_nupt_layout(self) -> bool:
        """
        Return True if root_dir contains Session* subdirectories
        (NUPT-FPV layout), False for the legacy flat layout.
        """
        entries = os.listdir(self.root_dir)
        return any(e.lower().startswith("session") for e in entries)

    # ------------------------------------------------------------------
    # Pair building — dispatcher
    # ------------------------------------------------------------------

    def _build_pairs(self):
        if self._is_nupt_layout():
            return self._build_pairs_nupt()
        else:
            return self._build_pairs_legacy()

    # ------------------------------------------------------------------
    # NUPT-FPV layout
    # ------------------------------------------------------------------

    def _build_pairs_nupt(self):
        """
        Walk Session*/Fingerprint/<subject>/ and Session*/FingerVein/<subject>/,
        pair images by (subject, session, sample_index), and return:
          samples   – list of (fp_path, vein_path, contiguous_label)
          label_map – dict  subject_str → contiguous 0-based int
        """
        # Collect all session directories
        session_dirs = sorted([
            os.path.join(self.root_dir, e)
            for e in os.listdir(self.root_dir)
            if e.lower().startswith("session")
            and os.path.isdir(os.path.join(self.root_dir, e))
        ])

        if not session_dirs:
            raise RuntimeError(f"No Session* folders found in {self.root_dir}")

        # fp_index[(subject, session, idx)] = path
        # vein_index[(subject, session, idx)] = path
        fp_index   = {}
        vein_index = {}

        for sess_dir in session_dirs:
            sess_name = os.path.basename(sess_dir)   # "Session1", "Session2", …

            # Find Fingerprint and FingerVein sub-dirs (case-insensitive)
            fp_base   = self._find_subdir(sess_dir, "print")
            vein_base = self._find_subdir(sess_dir, "vein")

            # Walk subject sub-directories
            for subject in sorted(os.listdir(fp_base)):
                subject_fp_dir = os.path.join(fp_base, subject)
                if not os.path.isdir(subject_fp_dir):
                    continue

                images = sorted(
                    f for f in os.listdir(subject_fp_dir)
                    if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))
                )
                for idx, img_name in enumerate(images):
                    key = (subject, sess_name, idx)
                    fp_index[key] = os.path.join(subject_fp_dir, img_name)

            for subject in sorted(os.listdir(vein_base)):
                subject_vein_dir = os.path.join(vein_base, subject)
                if not os.path.isdir(subject_vein_dir):
                    continue

                images = sorted(
                    f for f in os.listdir(subject_vein_dir)
                    if f.lower().endswith((".bmp", ".png", ".jpg", ".jpeg"))
                )
                for idx, img_name in enumerate(images):
                    key = (subject, sess_name, idx)
                    vein_index[key] = os.path.join(subject_vein_dir, img_name)

        # Pair on matching keys
        common_keys = sorted(set(fp_index) & set(vein_index))
        if not common_keys:
            raise RuntimeError(
                "No paired samples found in NUPT-FPV layout.\n"
                "Check that Fingerprint and FingerVein subject folders have "
                "the same names and the same number of images."
            )

        # Build contiguous label map from subject strings ("001", "002", …)
        subjects   = sorted(set(k[0] for k in common_keys))
        label_map  = {subj: idx for idx, subj in enumerate(subjects)}

        samples = [
            (fp_index[k], vein_index[k], label_map[k[0]])
            for k in common_keys
        ]

        print(
            f"[BiometricDataset] NUPT-FPV layout detected.\n"
            f"  Sessions  : {[os.path.basename(s) for s in session_dirs]}\n"
            f"  Subjects  : {len(subjects)}  "
            f"({subjects[0]}–{subjects[-1]})\n"
            f"  Pairs     : {len(samples)}"
        )
        return samples, label_map

    def _find_subdir(self, parent: str, keyword: str) -> str:
        """Return first child directory of parent whose name contains keyword."""
        for entry in os.listdir(parent):
            if keyword.lower() in entry.lower():
                path = os.path.join(parent, entry)
                if os.path.isdir(path):
                    return path
        raise FileNotFoundError(
            f"No sub-directory matching '{keyword}' in {parent}. "
            f"Found: {os.listdir(parent)}"
        )

    # ------------------------------------------------------------------
    # Legacy flat layout (kept for backward compatibility)
    # ------------------------------------------------------------------

    def _find_folder(self, keyword: str) -> str:
        for raw_entry in os.listdir(self.root_dir):
            clean_entry = _sanitize_name(raw_entry)
            if keyword.lower() in clean_entry.lower():
                raw_path   = os.path.join(self.root_dir, raw_entry)
                clean_path = os.path.join(self.root_dir, clean_entry)
                if raw_entry != clean_entry and os.path.exists(raw_path):
                    try:
                        os.rename(raw_path, clean_path)
                        print(f"[BiometricDataset] Renamed: {raw_entry!r} → {clean_entry!r}")
                    except OSError as exc:
                        print(f"[BiometricDataset] WARNING: could not rename: {exc}")
                        clean_path = raw_path
                return clean_path
        raise FileNotFoundError(
            f"No folder matching '{keyword}' found in {self.root_dir}."
        )

    def _build_pairs_legacy(self):
        fingerprint_dir = self._find_folder("print")
        fingervein_dir  = self._find_folder("vein")

        fp_map = {
            os.path.basename(p): p
            for p in glob.glob(os.path.join(fingerprint_dir, "**", "*.bmp"), recursive=True)
        }
        vein_map = {
            os.path.basename(p): p
            for p in glob.glob(os.path.join(fingervein_dir, "**", "*.bmp"), recursive=True)
            if "_segmented" not in p
        }

        if not fp_map:
            raise RuntimeError(f"No .bmp files under: {fingerprint_dir}")
        if not vein_map:
            raise RuntimeError(f"No .bmp files under: {fingervein_dir}")

        raw_pairs = []
        user_ids  = set()
        for fname in sorted(fp_map):
            if fname in vein_map:
                try:
                    parts      = fname.split("_")
                    user       = int(parts[0])
                    session    = int(parts[1])
                    finger_idx = int(parts[2].replace("f", "")) - 1
                except (ValueError, IndexError):
                    print(f"[BiometricDataset] WARNING: skipping '{fname}'")
                    continue
                user_ids.add(user)
                raw_pairs.append((fp_map[fname], vein_map[fname], user, session, finger_idx))

        if not raw_pairs:
            raise RuntimeError("No paired samples found in legacy layout.")

        if len(user_ids) > 1:
            raw_ids = [u for _, _, u, _, _ in raw_pairs]
            mode    = "real users"
        else:
            raw_ids = [fi * 100 + s for _, _, _, s, fi in raw_pairs]
            mode    = "pseudo-classes (single user)"

        unique_ids = sorted(set(raw_ids))
        label_map  = {uid: idx for idx, uid in enumerate(unique_ids)}
        samples    = [
            (fp, vn, label_map[rid])
            for (fp, vn, *_), rid in zip(raw_pairs, raw_ids)
        ]
        print(
            f"[BiometricDataset] Legacy layout detected. "
            f"{len(samples)} pairs across {len(label_map)} classes ({mode})."
        )
        return samples, label_map

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def train(self):
        """Enable augmentation (training mode)."""
        self.training_mode = True
        return self

    def eval(self):
        """Disable augmentation (eval mode)."""
        self.training_mode = False
        return self

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        fp_path, vein_path, label = self.samples[idx]

        fp_img = cv2.imread(fp_path)
        if fp_img is None:
            raise IOError(f"Cannot read fingerprint image: {fp_path}")
        fp_img = cv2.cvtColor(fp_img, cv2.COLOR_BGR2RGB)

        vein_img = cv2.imread(vein_path)
        if vein_img is None:
            raise IOError(f"Cannot read vein image: {vein_path}")
        vein_gray     = cv2.cvtColor(vein_img, cv2.COLOR_BGR2GRAY)
        vein_enhanced = self.clahe.apply(vein_gray)
        vein_img      = cv2.cvtColor(vein_enhanced, cv2.COLOR_GRAY2RGB)

        tfm         = self.transform if self.training_mode else self.val_transform
        fp_tensor   = tfm(fp_img)
        vein_tensor = tfm(vein_img)

        return fp_tensor, vein_tensor, torch.tensor(label, dtype=torch.long)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    root = r"D:\Intelligent_Biometric_System\NUPT-FPV-main\image"
    ds = BiometricDataset(root_dir=root)
    print(f"Total samples : {len(ds)}")
    print(f"Num classes   : {ds.num_classes}")
    fp, vn, lbl = ds[0]
    print(f"Fingerprint shape : {fp.shape}")
    print(f"Vein shape        : {vn.shape}")
    print(f"Label             : {lbl.item()}")