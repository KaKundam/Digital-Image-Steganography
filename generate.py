import os

print("=== GENERATE STEGO IMAGES ===")

print("\n[DCT]")
os.system("python -m methods.embed.DCT")

print("\n[DFT]")
os.system("python -m methods.embed.DFT")

print("\n[DWT]")
os.system("python -m methods.embed.DWT")

print("\n[Done] Generation completed.")
