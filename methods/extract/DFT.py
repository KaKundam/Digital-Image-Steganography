import cv2
import numpy as np
from ..utils import bits_to_text, DFT_MARKER, DFT_DELTA, BLOCK_SIZE

def extract_from_block(block):
    # Phải chuyển sang float để tính DFT chính xác
    dft = np.fft.fft2(block.astype(float))
    
    # Lấy biên độ tại (3,3) - Phải khớp với bên Embed
    val = np.abs(dft[3, 3])
    
    # Giải mã QIM
    k = round(val / DFT_DELTA)
    return str(k % 2)

def process_extract(channel):
    h, w = channel.shape
    h_crop = (h // BLOCK_SIZE) * BLOCK_SIZE
    w_crop = (w // BLOCK_SIZE) * BLOCK_SIZE
    
    extracted_bits = ""
    
    for i in range(0, h_crop, BLOCK_SIZE):
        for j in range(0, w_crop, BLOCK_SIZE):
            block = channel[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            extracted_bits += extract_from_block(block)
            
            # Check DFT_MARKER mỗi khi đủ 1 ký tự (8 bits) để tối ưu tốc độ
            if len(extracted_bits) % 8 == 0:
                # Chỉ kiểm tra đoạn cuối cùng có độ dài bằng DFT_MARKER
                # Ví dụ DFT_MARKER là @@@@@ (40 bits), chỉ lấy 40 bit cuối để check text
                try:
                    current_text = bits_to_text(extracted_bits)
                    if current_text.endswith(DFT_MARKER):
                        return current_text[:-len(DFT_MARKER)] # Cắt bỏ DFT_MARKER và trả về tin
                except:
                    # Bỏ qua lỗi nếu bits chưa tạo thành ký tự utf-8 hợp lệ
                    pass
    
    return "[Fail] Không tìm thấy DFT_MARKER (Tin nhắn có thể đã bị hỏng do nén/nhiễu)"

def extract_dft(stego_image_path, mode='blue'): # Default nên để giống embed
    # ... (Phần đọc ảnh giữ nguyên như code của bạn) ...
    if mode == 'gray':
        img = cv2.imread(stego_image_path, cv2.IMREAD_GRAYSCALE)
        target = img
    elif mode == 'blue':
        img = cv2.imread(stego_image_path)
        if img is None: return None
        b, g, r = cv2.split(img)
        target = b
    elif mode == 'cb':
        img = cv2.imread(stego_image_path)
        if img is None: return None
        img_ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(img_ycrcb)
        target = cb
    else: return "Lỗi Mode"

    if target is None: return "Lỗi đọc file."

    return process_extract(target)

if __name__ == "__main__":
    import os
    import glob

    STEGO_FOLDER = "output/dft"
    MODE = "blue"   # phải trùng với mode lúc embed

    types = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif')
    stego_files = []
    for t in types:
        stego_files.extend(glob.glob(os.path.join(STEGO_FOLDER, t)))

    if not stego_files:
        print("[!] Không tìm thấy ảnh stego")
        exit()

    print(f"[+] DFT extracting: {len(stego_files)} ảnh")
    print("-" * 50)

    for img_path in stego_files:
        name = os.path.basename(img_path)
        print(f"[*] {name}")

        msg = extract_dft(img_path, mode=MODE)

        print(f"    -> Extracted message: {msg}")
        print("-" * 50)
