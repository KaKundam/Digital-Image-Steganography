# Digital Image Steganography (DFT & DWT)

## 1. Requirements

- Python **3.9+**
- OS: Windows / Linux / macOS

### Required Libraries
```bash
pip install -r requirement.txt
````

---

## 2. Project Structure

```text
Digital-Image-Steganography/
│
├── dataset/                # Original cover images
│   ├── baboon.png
│   └── lena.png
│
├── methods/
│   ├── embed/
│   │   ├── DFT.py
│   │   └── DWT.py
│   ├── extract/
│   │   ├── DFT.py
│   │   └── DWT.py
│
├── output/
│   ├── dft/                # Stego images (DFT)
│   └── dwt/                # Stego images (DWT)
│
├── evaluate.py             # PSNR / SSIM / BER evaluation
└── README.md
```

---

## 3. Embedding (Generate Stego Images)

### DFT-based Steganography

```bash
python -m methods.embed.DFT
```

### DWT-based Steganography

```bash
python -m methods.embed.DWT
```

### One For All

```bash
python generate.py
```

Stego images will be saved to:

* `output/dft/`
* `output/dwt/`

---

## 4. Extraction (Recover Secret Message)

### DFT

```bash
python -m methods.extract.DFT
```

### DWT

```bash
python -m methods.extract.DWT
```

The extracted message will be printed to the terminal.

---

## 5. Evaluation (Quality & Robustness Testing)

```bash
python evaluate.py
```

### Metrics

* **MSE** (Mean Squared Error)
* **PSNR** (Peak Signal-to-Noise Ratio)
* **SSIM** (Structural Similarity Index)
* **BER** (Bit Error Rate)

### Test Scenarios

* No Attack
* Gaussian Noise Attack
* JPEG Compression (Quality = 90, 80)

---

## 6. Important Notes

* `dataset/` must contain **only original images**
* `output/` can be safely deleted and regenerated
* To debug extraction:

  * Set `ENABLE_EXTRACT_TEST = True` in the embedding scripts

---

## 7. Quick Pipeline (TL;DR)

```bash
python generate.py
python evaluate.py
```
