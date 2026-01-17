import cv2
import numpy as np
import pywt
import os
import glob
from ..utils import text_to_bits

CHANNEL = 0  # Grayscale or single channel

def dwt2(channel):
    return pywt.dwt2(channel, 'haar')

def idwt2(coeffs):
    return pywt.idwt2(coeffs, 'haar')

def embed_bits_in_HH(HH, binary_message):
    flat = HH.flatten().copy()

    msg_len = len(binary_message)
    header = format(msg_len, '032b')
    payload = header + binary_message

    if len(payload) > len(flat):
        raise ValueError("Message too long for image")

    for i, bit in enumerate(payload):
        val = int(flat[i])
        flat[i] = float((val & ~1) | int(bit))

    return flat.reshape(HH.shape)

def embed_dwt(image_path, message, output_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("Cannot read image")

    coeffs = dwt2(img)
    LL, (LH, HL, HH) = coeffs

    bits = text_to_bits(message)
    HH_new = embed_bits_in_HH(HH, bits)

    stego = idwt2((LL, (LH, HL, HH_new)))
    stego = np.clip(stego, 0, 255).astype(np.uint8)

    cv2.imwrite(output_path, stego)

# ================= CLI =================

if __name__ == "__main__":
    INPUT = "dataset"
    OUTPUT = "output/dwt"
    MESSAGE = "DigitalForensic2026"

    os.makedirs(OUTPUT, exist_ok=True)
    images = glob.glob(os.path.join(INPUT, "*.png"))

    print(f"[+] DWT embedding: {len(images)} images")
    print("-" * 50)

    for img in images:
        name = os.path.basename(img)
        out = os.path.join(OUTPUT, name.replace(".png", "_output_DWT.png"))
        embed_dwt(img, MESSAGE, out)
        print(f"[*] {name} -> Saved: {out}")
