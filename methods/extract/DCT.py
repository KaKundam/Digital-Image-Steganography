import cv2
import numpy as np
import os
from ..utils import bits_to_text, DCT_DELTA, BLOCK_SIZE, DFT_MARKER

CHANNEL = 0       # Blue
COEF_POS = (4, 3) # Phải khớp với Embed

def extract_dct(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return "[ERROR] Cannot read image"

    h, w, _ = img.shape
    h_crop = (h // BLOCK_SIZE) * BLOCK_SIZE
    w_crop = (w // BLOCK_SIZE) * BLOCK_SIZE

    # Lấy kênh Blue
    channel = img[:h_crop, :w_crop, CHANNEL]

    extracted_bits = ""

    for i in range(0, h_crop, BLOCK_SIZE):
        for j in range(0, w_crop, BLOCK_SIZE):
            block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            
            # DCT
            dct_block = cv2.dct(block.astype(np.float32))
            
            # Lấy giá trị
            u, v = COEF_POS
            val = dct_block[u, v]
            
            # Giải mã QIM: k = round(val / DELTA)
            # Bit = k % 2
            k = round(val / DCT_DELTA)
            extracted_bits += str(k % 2)

            # Check DFT_MARKER (Logic tối ưu)
            if len(extracted_bits) % 8 == 0:
                try:
                    # Chỉ check đoạn cuối
                    current_text = bits_to_text(extracted_bits)
                    if current_text.endswith(DFT_MARKER):
                        return current_text[:-len(DFT_MARKER)]
                except:
                    pass

    return "[WARN] DFT_MARKER not found (Tin có thể bị hỏng)"

if __name__ == "__main__":
    pass

if __name__ == "__main__":
    INPUT_FOLDER = "output/dct"

    images = glob.glob(os.path.join(INPUT_FOLDER, "*.png"))

    print(f"[+] DCT extracting: {len(images)} ảnh")
    print("-" * 50)

    for img in images:
        msg = extract_dct(img)
        print(f"{os.path.basename(img)} -> {msg}")
