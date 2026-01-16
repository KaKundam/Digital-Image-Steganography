import cv2
import numpy as np
from utils import bits_to_text, MARKER, DELTA, BLOCK_SIZE

def extract_from_block(block):
    dft = np.fft.fft2(block.astype(float))
    # Lấy biên độ tại (3,3)
    val = np.abs(dft[3, 3])
    
    # Giải mã QIM
    k = round(val / DELTA)
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
            
            # Check Marker
            if len(extracted_bits) % 8 == 0 and len(extracted_bits) >= 24:
                # Chỉ check đoạn cuối để tối ưu
                suffix = extracted_bits[-24:]
                try:
                    if bits_to_text(suffix) == MARKER:
                        return bits_to_text(extracted_bits).split(MARKER)[0]
                except:
                    pass
    
    # Thử check lần cuối nếu vòng lặp kết thúc mà chưa return
    try:
        full_msg = bits_to_text(extracted_bits)
        if MARKER in full_msg:
            return full_msg.split(MARKER)[0]
    except:
        pass
        
    return None

def extract_dft(stego_image_path, mode='gray'):
    # Đọc ảnh đúng chuẩn
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

    msg = process_extract(target)
    if msg:
        return msg
    else:
        return "[Fail] Vẫn không tìm thấy (Do Delta chưa đủ lớn hoặc ảnh quá nhiễu)."