import cv2
import numpy as np
import os
import glob

def text_to_bits(text):
    return ''.join(format(ord(c), '08b') for c in text)

def ber(original_bits, extracted_bits):
    errors = sum(o != e for o, e in zip(original_bits, extracted_bits))
    return errors / len(original_bits)

class IWTSteganography:
    def __init__(self):
        self.channel_to_use = 0
        self.DELIMITER = '1111111111111110'

    def _text_to_binary(self, text):
        binary_str = ''.join(format(ord(char), '08b') for char in text)
        return binary_str + self.DELIMITER

    def _binary_to_text(self, binary_str):
        chars = []
        for i in range(0, len(binary_str), 8):
            byte = binary_str[i:i + 8]
            if len(byte) == 8:
                chars.append(chr(int(byte, 2)))
        return ''.join(chars)

    def _iwt_haar_forward(self, block):
        block = block.astype(np.int32)
        even = block[0::2]
        odd = block[1::2]
        d = odd - even
        s = even + (d // 2)
        return s, d

    def _iwt_haar_inverse(self, s, d):
        even = s - (d // 2)
        odd = d + even
        reconstructed = np.zeros(len(s) + len(d), dtype=np.int32)
        reconstructed[0::2] = even
        reconstructed[1::2] = odd
        return reconstructed

    def embed_data(self, image_path, secret_data, output_folder="output_DWT"):
        img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"[Lỗi] Không đọc được ảnh: {image_path}")
            return None

        if img.ndim == 2:
            is_grayscale = True
            height, width = img.shape
            channel_data = img
        else:
            is_grayscale = False
            height, width, channels = img.shape
            channel_data = img[:, :, self.channel_to_use]

        if height % 2 != 0: height -= 1
        if width % 2 != 0: width -= 1

        channel_data = channel_data[:height, :width]

        binary_secret = self._text_to_binary(secret_data)
        data_len = len(binary_secret)

        flat_pixels = channel_data.flatten()
        s, d = self._iwt_haar_forward(flat_pixels)

        if data_len > len(d):
            print(f"[Lỗi] Ảnh {os.path.basename(image_path)} quá nhỏ!")
            return None

        d_modified = d.copy()
        for i in range(data_len):
            d_modified[i] = (d_modified[i] & ~1) | int(binary_secret[i])

        reconstructed_flat = self._iwt_haar_inverse(s, d_modified)
        reconstructed_channel = reconstructed_flat.reshape((height, width))
        reconstructed_channel = np.clip(reconstructed_channel, 0, 255).astype(np.uint8)

        if is_grayscale:
            stego_img = reconstructed_channel
        else:
            stego_img = img[:height, :width, :].copy()
            stego_img[:, :, self.channel_to_use] = reconstructed_channel

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        filename = os.path.basename(image_path)
        name, ext = os.path.splitext(filename)

        # --- THAY ĐỔI Ở ĐÂY: Đổi tên file output thành _output_DWT ---
        output_path = os.path.join(output_folder, f"{name}_output_DWT.png")

        cv2.imwrite(output_path, stego_img)
        return output_path

    def extract_data(self, stego_image_path):
        img = cv2.imread(stego_image_path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return "Lỗi đọc file"

        if img.ndim == 2:
            channel_data = img
        else:
            channel_data = img[:, :, self.channel_to_use]

        flat_pixels = channel_data.flatten()
        s, d = self._iwt_haar_forward(flat_pixels)

        binary_data = ""
        delimiter_found = False

        for val in d:
            binary_data += str(val & 1)
            if binary_data.endswith(self.DELIMITER):
                binary_data = binary_data[:-len(self.DELIMITER)]
                delimiter_found = True
                break

        if not delimiter_found:
            return "[Cảnh báo] Không tìm thấy dấu hiệu kết thúc tin nhắn."

        return self._binary_to_text(binary_data)

def embed_dwt(image_path, secret_text, output_path):
    tool = IWTSteganography()
    folder = os.path.dirname(output_path)
    os.makedirs(folder, exist_ok=True)
    return tool.embed_data(image_path, secret_text, folder)

def extract_dwt(stego_image_path):
    tool = IWTSteganography()
    return tool.extract_data(stego_image_path)


if __name__ == "__main__":
    import glob
    for img_path in glob.glob("output/dwt/*.png"):
        print(f"[*] {os.path.basename(img_path)}")
        print(f"    -> Extracted message: {extract_dwt(img_path)}")


