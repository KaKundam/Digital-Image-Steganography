import cv2
import numpy as np
import os
import glob
# Import DFT_MARKER và DCT_DELTA từ utils để đồng bộ
from ..utils import text_to_bits, DCT_DELTA, BLOCK_SIZE, DFT_MARKER 

CHANNEL = 0          # Kênh Blue
COEF_POS = (4, 3)    # Tần số trung bình (Mid-frequency)

def embed_dct_block(block, bit):
    # 1. DCT (chuyển sang float32)
    dct_block = cv2.dct(block.astype(np.float32))
    
    # 2. Lấy giá trị hệ số
    u, v = COEF_POS
    val = dct_block[u, v]
    target_bit = int(bit)
    
    # 3. QIM (Quantization Index Modulation)
    # Tìm số nguyên k sao cho k * DELTA gần val nhất
    k = round(val / DCT_DELTA)
    
    # Điều chỉnh k chẵn/lẻ theo bit
    if k % 2 != target_bit:
        if (val - k * DCT_DELTA) >= 0: k += 1
        else: k -= 1
            
    # Gán giá trị mới
    dct_block[u, v] = k * DCT_DELTA
    
    # 4. IDCT
    return cv2.idct(dct_block)

def embed_dct(image_path, secret_text, output_folder):
    img = cv2.imread(image_path) # Mặc định đọc BGR
    if img is None:
        print(f"[ERROR] Cannot read {image_path}")
        return None

    # --- QUAN TRỌNG: Thêm DFT_MARKER ---
    full_text = secret_text + DFT_MARKER
    bits = text_to_bits(full_text)
    # -------------------------------

    h, w, _ = img.shape
    # Crop ảnh cho chẵn block
    h_crop = (h // BLOCK_SIZE) * BLOCK_SIZE
    w_crop = (w // BLOCK_SIZE) * BLOCK_SIZE
    
    img_cropped = img[:h_crop, :w_crop].copy()
    channel = img_cropped[:, :, CHANNEL] # Lấy kênh Blue

    bit_idx = 0
    max_bits = (h_crop * w_crop) // (BLOCK_SIZE * BLOCK_SIZE)
    
    if len(bits) > max_bits:
        print(f"[WARN] Message too long. Truncating to {max_bits} bits.")
        bits = bits[:max_bits]

    # Duyệt block
    for i in range(0, h_crop, BLOCK_SIZE):
        for j in range(0, w_crop, BLOCK_SIZE):
            if bit_idx >= len(bits): break

            block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            
            # Nhúng QIM
            modified_block = embed_dct_block(block, bits[bit_idx])
            
            # Gán lại (Clip để đảm bảo không tràn pixel)
            channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = np.clip(modified_block, 0, 255)
            bit_idx += 1

    # G ghép lại vào ảnh gốc
    img_cropped[:, :, CHANNEL] = channel.astype(np.uint8)
    
    final_img = img.copy()
    final_img[:h_crop, :w_crop] = img_cropped

    os.makedirs(output_folder, exist_ok=True)
    name = os.path.splitext(os.path.basename(image_path))[0]
    output_path = os.path.join(output_folder, f"{name}_stego_dct.png")

    # Lưu PNG để tránh nén lossy
    cv2.imwrite(output_path, final_img)
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
