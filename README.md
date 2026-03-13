# Intelligent Biometric System

This project is an **Intelligent Biometric System** that performs identity verification by combining two different biometric modalities: **fingerprints** and **finger-veins** (a "multimodal" approach). By fusing information from both the surface (fingerprint) and subsurface (vein) of the finger, the system becomes highly accurate and resistant to spoofing.

The system features:
- **Multimodal Fusion Engine**: Combines pairs of fingerprint and finger-vein images.
- **Advanced Architecture**: Uses two pre-trained `EfficientNet` backbones for feature extraction and a Vision Transformer (`ViT`) for intelligent modal fusion.
- **Robust Metric Learning**: Trained with a combination of 5 custom loss functions (ArcFace, Triplet Loss, Supervised Contrastive, Verification Loss, and Intra-class Variance) to maximize inter-class separation and minimize intra-class distances.

## 🛠️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Mrutyunjaya98Gouda/Intelligent_Biometric_System.git
   cd Intelligent_Biometric_System
   ```

2. **Set up a Python Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   Make sure to install the required PyTorch packages along with standard computer vision and data science libraries:
   ```bash
   pip install torch torchvision torchaudio timm opencv-python matplotlib numpy scikit-learn
   ```

## 📂 Data Preparation

The `BiometricDataset` module natively supports the **NUPT-FPV** dataset folder structure, but can also work with a simple legacy layout. 

**Important Config Update:**
Before running, you must open `demo.py` and update the **hardcoded paths** at the top of the file to point to your local dataset and desired output directories:
```python
# In demo.py (Lines 48-52)
DATASET_ROOT  = r"path/to/NUPT-FPV-main/image"
MODEL_PATH    = r"path/to/save/best_biometric_model.pth"
OUTPUT_DIR    = r"path/to/save/test_results"
LOSS_CSV_PATH = r"path/to/save/loss_curves.csv"
```

## 🚀 How to Run

The easiest way to interact with the system is through `demo.py`, which provides both an interactive menu and command-line arguments.

### Option A: Interactive Menu
Simply run the script with no arguments to get an interactive menu guiding you through training, evaluation, and verification:
```bash
python demo.py
```

### Option B: Command Line Automation

**1. Train the model:**
```bash
python demo.py train
```
*This will train the model, save `best_biometric_model.pth`, and generate `loss_curves.png`.*

**2. Evaluate the model:**
```bash
python demo.py eval
```
*This evaluates the trained model on the dataset and generates a detailed similarity heatmap and CSV metrics in the `test_results` directory.*

**3. Train and then evaluate:**
```bash
python demo.py train_eval
```

**4. Quick Verification (Test two identities):**
Test if Person A and Person B are the same identity using a fixed threshold inference:
```bash
python demo.py verify_quick \
    --fp1 "path/to/person1_fingerprint.bmp" \
    --vein1 "path/to/person1_vein.bmp" \
    --fp2 "path/to/person2_fingerprint.bmp" \
    --vein2 "path/to/person2_vein.bmp"
```

**5. Full Verification (Calibrated threshold):**
Run a full verification test that calibrates the threshold directly from your dataset distribution and evaluates a genuine pairing vs. an impostor pairing:
```bash
python demo.py verify \
    --fp1    "Session1/Fingerprint/001/001_1.bmp" \
    --vein1  "Session1/FingerVein/001/001_1.bmp"  \
    --fp1b   "Session2/Fingerprint/001/001_1.bmp" \
    --vein1b "Session2/FingerVein/001/001_1.bmp"  \
    --fp2    "Session1/Fingerprint/002/002_1.bmp" \
    --vein2  "Session1/FingerVein/002/002_1.bmp"
```
