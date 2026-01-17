import cv2
import numpy as np
import pywt
import os
import glob
from ..utils import bits_to_text

def dwt2(channel):
    return pywt.dwt2(channel, 'haar')

def extract_bits_from_HH(HH):
    flat = HH.flatten()

    # --- Read header ---
    header = ""
    for i in range(32):
        header += str(int(flat[i]) & 1)

    try:
        msg_len = int(header, 2)
    except:
        return "[ERROR] Invalid header"

    # --- SAFETY CHECK ---
    max_bits = len(flat) - 32
    msg_len = min(msg_len, max_bits)

    bits = ""
    for i in range(32, 32 + msg_len):
        bits += str(int(flat[i]) & 1)

    return bits_to_text(bits)


def extract_dwt(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "[ERROR] Cannot read image"

    coeffs = dwt2(img)
    _, (_, _, HH) = coeffs

    return extract_bits_from_HH(HH)

# ================= CLI =================

if __name__ == "__main__":
    INPUT = "output/dwt"
    images = glob.glob(os.path.join(INPUT, "*.png"))

    print(f"[+] DWT extracting: {len(images)} images")
    print("-" * 50)

    for img in images:
        msg = extract_dwt(img)
        print(f"{os.path.basename(img)} -> {msg}")
