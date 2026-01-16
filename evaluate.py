import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
from math import log10
from methods.utils import text_to_bits, bits_to_text

# ==============================
# IMPORT EXTRACTORS
# ==============================
from methods.extract.DFT import extract_dft
from methods.extract.DWT import extract_dwt

# ==============================
# CONFIG
# ==============================
DATASET_DIR = "dataset"
OUTPUT_DIR = "output"

DFT_DIR = os.path.join(OUTPUT_DIR, "dft")
DWT_DIR = os.path.join(OUTPUT_DIR, "dwt")

SECRET_MESSAGE = "DigitalForensic2026"

JPEG_QUALITIES = [90, 80]
GAUSSIAN_SIGMA = 10

# ==============================
# METRICS
# ==============================
def compute_mse(img1, img2):
    return np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)

def compute_psnr(mse):
    if mse == 0:
        return float('inf')
    return 10 * log10((255 ** 2) / mse)

def compute_ssim(img1, img2):
    return ssim(img1, img2, data_range=255)

def compute_ber(original_text, extracted_text):
    if not extracted_text:
        return 1.0 # Mất hoàn toàn

    orig_bits = text_to_bits(original_text)
    
    try:
        ext_bits = text_to_bits(extracted_text)
    except:
        return 1.0 # Lỗi convert do ký tự rác

    # Chỉ so sánh trong phạm vi độ dài của tin gốc
    n = len(orig_bits)
    m = len(ext_bits)
    
    # Nếu trích xuất được quá ít -> coi phần thiếu là lỗi
    # Nếu trích xuất được quá nhiều (rác) -> cắt bớt để so sánh
    compare_len = min(n, m)
    
    errors = sum(orig_bits[i] != ext_bits[i] for i in range(compare_len))
    
    # Cộng thêm số bit bị thiếu
    errors += (n - compare_len)
    
    return errors / n

# ==============================
# ATTACKS
# ==============================
def gaussian_noise(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def jpeg_compress(img, quality):
    tmp = "temp.jpg"
    cv2.imwrite(tmp, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    compressed = cv2.imread(tmp, cv2.IMREAD_GRAYSCALE)
    os.remove(tmp)
    return compressed

# ==============================
# EVALUATION
# ==============================
def evaluate_method(method_name, stego_dir, extractor):
    print(f"\n=== {method_name} Evaluation ===")

    for filename in os.listdir(stego_dir):
        if not filename.endswith(".png"):
            continue

        stego_path = os.path.join(stego_dir, filename)
        original_path = os.path.join(DATASET_DIR, filename.split("_")[0] + ".png")

        original = cv2.imread(original_path, cv2.IMREAD_GRAYSCALE)
        stego = cv2.imread(stego_path, cv2.IMREAD_GRAYSCALE)

        print(f"\nImage: {filename}")

        # -------- Scenario 1: No Attack --------
        mse = compute_mse(original, stego)
        psnr = compute_psnr(mse)
        ssim_val = compute_ssim(original, stego)

        extracted = extractor(stego_path)
        ber_no_attack = compute_ber(SECRET_MESSAGE, extracted)

        print("  [No Attack]")
        print(f"    MSE  : {mse:.4f}")
        print(f"    PSNR : {psnr:.2f} dB")
        print(f"    SSIM : {ssim_val:.4f}")
        print(f"    BER  : {ber_no_attack:.4f}")

        # -------- Scenario 2: Gaussian Noise --------
        noisy = gaussian_noise(stego, GAUSSIAN_SIGMA)
        cv2.imwrite("temp_noise.png", noisy)
        extracted_noise = extractor("temp_noise.png")
        ber_noise = compute_ber(SECRET_MESSAGE, extracted_noise)

        print("  [Gaussian Noise]")
        print(f"    BER  : {ber_noise:.4f}")

        # -------- Scenario 3: JPEG Compression --------
        for q in JPEG_QUALITIES:
            compressed = jpeg_compress(stego, q)
            cv2.imwrite("temp_jpeg.png", compressed)
            extracted_jpeg = extractor("temp_jpeg.png")
            ber_jpeg = compute_ber(SECRET_MESSAGE, extracted_jpeg)

            print(f"  [JPEG Q={q}]")
            print(f"    BER  : {ber_jpeg:.4f}")

    # Cleanup
    for f in ["temp_noise.png", "temp_jpeg.png"]:
        if os.path.exists(f):
            os.remove(f)

# ==============================
# MAIN
# ==============================
if __name__ == "__main__":
    print("===== DIGITAL IMAGE STEGANOGRAPHY EVALUATION =====")

    evaluate_method(
        method_name="DFT",
        stego_dir=DFT_DIR,
        extractor=extract_dft
    )

    evaluate_method(
        method_name="DWT",
        stego_dir=DWT_DIR,
        extractor=extract_dwt
    )

    print("\n[Done] Evaluation completed.")
