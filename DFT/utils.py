import numpy as np

# Cấu hình "Nồi đồng cối đá" để bao chạy
MARKER = "###"
DELTA = 60       # Tăng Delta lên cao để chống nhiễu do cắt cụt pixel (Clipping)
BLOCK_SIZE = 8   

def text_to_bits(text):
    full_text = text + MARKER
    bits = bin(int.from_bytes(full_text.encode('utf-8'), 'big'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

def bits_to_text(bits):
    try:
        n = int(bits, 2)
        return n.to_bytes((n.bit_length() + 7) // 8, 'big').decode('utf-8', errors='ignore')
    except:
        return ""