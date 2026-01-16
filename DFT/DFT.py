# DFT.py
import os
from dft_stego import embed_dft
from dft_extract import extract_dft

def main():
    print("=== DFT Steganography Multi-Mode ===")
    print("1. Nhúng tin (Embed)")
    print("2. Trích xuất tin (Extract)")
    
    choice = input("Chọn chức năng (1/2): ").strip()

    if choice in ['1', '2']:
        print("\n--- Chọn chế độ ảnh ---")
        print("a. Gray Image (Ảnh xám)")
        print("b. Color Image - RGB (Nhúng vào kênh Blue)")
        print("c. Color Image - YCbCr (Nhúng vào kênh Cb)")
        mode_input = input("Chọn chế độ (a/b/c): ").strip().lower()
        
        mode_map = {'a': 'gray', 'b': 'rgb_blue', 'c': 'ycc_cb'}
        selected_mode = mode_map.get(mode_input)
        
        if not selected_mode:
            print("Chế độ không hợp lệ!")
            return

        if choice == '1':
            input_path = input("\nNhập đường dẫn ảnh gốc: ").strip()
            if not os.path.exists(input_path): return
            text = input("Nhập nội dung cần giấu: ").strip()
            output_path = input("Nhập tên file đầu ra (VD: out.png): ").strip()
            
            embed_dft(input_path, text, output_path, mode=selected_mode)

        elif choice == '2':
            stego_path = input("\nNhập đường dẫn ảnh Stego: ").strip()
            if not os.path.exists(stego_path): return
            
            print(f"Đang trích xuất với chế độ {selected_mode}...")
            msg = extract_dft(stego_path, mode=selected_mode)
            print("-" * 30)
            print("KẾT QUẢ:", msg)
            print("-" * 30)

if __name__ == "__main__":
    main()