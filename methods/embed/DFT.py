import cv2
import numpy as np
from ..utils import text_to_bits, DFT_DELTA, DFT_MARKER, BLOCK_SIZE

def embed_block(block, bit):
    """Nhúng 1 bit vào khối 8x8 dựa trên Biên độ (Magnitude)"""
    # 1. DFT
    dft = np.fft.fft2(block.astype(float))
    
    # 2. Tách Biên độ và Pha
    magnitude = np.abs(dft)
    phase = np.angle(dft)
    
    # Chọn tần số (3,3) và đối xứng (5,5)
    u, v = 3, 3
    sym_u, sym_v = BLOCK_SIZE - u, BLOCK_SIZE - v
    
    # 3. QIM trên Biên độ (Magnitude)
    val = magnitude[u, v]
    target_bit = int(bit)
    
    # Tìm bội số k của DELTA
    k = round(val / DFT_DELTA)
    if k % 2 != target_bit:
        # Điều chỉnh k để khớp với bit (chẵn/lẻ)
        if (val - k * DFT_DELTA) >= 0: k += 1
        else: k -= 1
    
    new_val = k * DFT_DELTA
    
    # 4. Cập nhật Biên độ (Giữ nguyên Pha) cho cả 2 điểm đối xứng
    magnitude[u, v] = new_val
    magnitude[sym_u, sym_v] = new_val
    
    # 5. Tái tạo số phức từ Biên độ mới + Pha cũ
    # Z = R * exp(j * phi)
    dft_new = magnitude * np.exp(1j * phase)
    
    # 6. IDFT
    return np.real(np.fft.ifft2(dft_new))

def process_channel(channel, bits):
    h, w = channel.shape
    # Cắt ảnh cho chẵn block 8
    h_crop = (h // BLOCK_SIZE) * BLOCK_SIZE
    w_crop = (w // BLOCK_SIZE) * BLOCK_SIZE
    
    channel_cropped = channel[:h_crop, :w_crop].copy()
    bit_idx = 0
    
    print(f"Tiến hành nhúng {len(bits)} bits vào ảnh kích thước {w_crop}x{h_crop}...")
    
    for i in range(0, h_crop, BLOCK_SIZE):
        for j in range(0, w_crop, BLOCK_SIZE):
            if bit_idx >= len(bits): break
            
            block = channel_cropped[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            modified_block = embed_block(block, bits[bit_idx])
            
            # Gán lại
            channel_cropped[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = modified_block
            bit_idx += 1
            
    final_channel = channel.copy()
    final_channel[:h_crop, :w_crop] = channel_cropped
    return final_channel

def embed_dft(image_path, secret_text, output_path, mode='blue'):
    img = cv2.imread(image_path)
    if img is None:
        print("Lỗi: Không tìm thấy ảnh.")
        return False

    full_text = secret_text + DFT_MARKER 
    bits = text_to_bits(full_text)

    # Xử lý các Mode
    if mode == 'gray':
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        res = process_channel(img, bits)
        # Ép kiểu an toàn
        final_img = np.clip(res, 0, 255).astype(np.uint8)

    elif mode == 'blue':
        b, g, r = cv2.split(img)
        res_b = process_channel(b, bits)
        final_img = cv2.merge([np.clip(res_b, 0, 255).astype(np.uint8), g, r])

    elif mode == 'cb':
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(img_ycrcb)
        res_cb = process_channel(cb, bits)
        final_img = cv2.merge([y, cr, np.clip(res_cb, 0, 255).astype(np.uint8)])
        final_img = cv2.cvtColor(final_img, cv2.COLOR_YCrCb2BGR)
    else:
        return False

    if not output_path.lower().endswith(".png"):
        output_path += ".png"
    
    cv2.imwrite(output_path, final_img)
    print(f"[Success] Đã lưu: {output_path} (Delta={DFT_DELTA})")
    return True

if __name__ == "__main__":
    import os
    import glob

    INPUT_FOLDER = "dataset"
    OUTPUT_FOLDER = "output/dft"
    SECRET_MESSAGE = "DigitalForensic2026"
    MODE = "blue"          # gray | blue | cb

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    types = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif')
    image_files = []
    for t in types:
        image_files.extend(glob.glob(os.path.join(INPUT_FOLDER, t)))

    if not image_files:
        print(image_files)
        print("[!] Không tìm thấy ảnh input")
        exit()

    print(f"[+] DFT embedding: {len(image_files)} ảnh")
    print("-" * 50)

    for img_path in image_files:
        name = os.path.splitext(os.path.basename(img_path))[0]
        output_path = os.path.join(
            OUTPUT_FOLDER,
            f"{name}_stego_dft.png"
        )

        print(f"[*] {name}")

        ok = embed_dft(
            img_path,
            SECRET_MESSAGE,
            output_path,
            mode=MODE
        )

        if ok:
            print(f"    -> Saved: {output_path}")

        print("-" * 50)