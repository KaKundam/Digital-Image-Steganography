import cv2
import numpy as np
import os
import glob
from ..utils import text_to_bits, bits_to_text, DCT_DELTA, BLOCK_SIZE

DELIMITER = '1111111111111110'
CHANNEL = 0          # Blue channel
COEF_POS = (4, 3)    # Mid-frequency DCT coefficient

def dct2(block):
    return cv2.dct(block.astype(np.float32))

def idct2(block):
    return cv2.idct(block)

def embed_dct(image_path, secret_text, output_folder):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        print(f"[ERROR] Cannot read {image_path}")
        return None

    h, w, _ = img.shape
    h -= h % BLOCK_SIZE
    w -= w % BLOCK_SIZE

    img = img[:h, :w]
    channel = img[:, :, CHANNEL].astype(np.float32)

    bits = text_to_bits(secret_text)
    bit_idx = 0

    for i in range(0, h, BLOCK_SIZE):
        for j in range(0, w, BLOCK_SIZE):
            if bit_idx >= len(bits):
                break

            block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            dct_block = dct2(block)

            u, v = COEF_POS
            coef = int(dct_block[u, v])
            # dct_block[u, v] = (int(dct_block[u, v]) & ~1) | int(bits[bit_idx])

            bit = int(bits[bit_idx])

            if bit == '0':
                coef = coef & ~1
            else:
                coef = coef | 1

            dct_block[u, v] = coef
            bit_idx += 1

            channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = idct2(dct_block)

    channel = np.clip(channel, 0, 255).astype(np.uint8)
    img[:, :, CHANNEL] = channel

    os.makedirs(output_folder, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{name}_stego_dct.png")

    cv2.imwrite(output_path, img)
    return output_path


if __name__ == "__main__":
    INPUT_FOLDER = "dataset"
    OUTPUT_FOLDER = "output/dct"
    SECRET_MESSAGE = "DigitalForensic2026"

    images = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif"):
        images.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    print(f"[+] DCT embedding: {len(images)} ảnh")
    print("-" * 50)

    for img in images:
        print(f"[*] {os.path.basename(img)}")
        out = embed_dct(img, SECRET_MESSAGE, OUTPUT_FOLDER)
        if out:
            print(f"    -> Saved: {out}")
        print("-" * 50)
