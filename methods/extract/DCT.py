import cv2
import numpy as np
import os
import glob
from ..utils import text_to_bits, bits_to_text, DCT_DELTA, BLOCK_SIZE


DELIMITER = '1111111111111110'
CHANNEL = 0
BLOCK_SIZE = 8
COEF_POS = (4, 3)

def dct2(block):
    return cv2.dct(block.astype(np.float32))

def extract_dct(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return "[ERROR] Cannot read image"

    h, w, _ = img.shape
    h -= h % BLOCK_SIZE
    w -= w % BLOCK_SIZE

    channel = img[:h, :w, CHANNEL].astype(np.float32)

    bits = ""

    for i in range(0, h, BLOCK_SIZE):
        for j in range(0, w, BLOCK_SIZE):
            block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            dct_block = dct2(block)

            u, v = COEF_POS
            bits += str(int(dct_block[u, v]) & 1)

            if bits.endswith(DELIMITER):
                bits = bits[:-len(DELIMITER)]
                return bits_to_text(bits)

    return "[WARN] Delimiter not found"


if __name__ == "__main__":
    INPUT_FOLDER = "output/dct"

    images = glob.glob(os.path.join(INPUT_FOLDER, "*.png"))

    print(f"[+] DCT extracting: {len(images)} ảnh")
    print("-" * 50)

    for img in images:
        msg = extract_dct(img)
        print(f"{os.path.basename(img)} -> {msg}")
