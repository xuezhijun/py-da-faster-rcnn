import os
from PIL import Image

for img_file in os.listdir():
    if img_file == "png_to_jpg_converter.py":
        continue
    try:
        print(img_file)
        img = Image.open(img_file)
        rgb_im = img.convert('RGB')
        rgb_im.save(img_file[:-3] + "jpg")
    except:
        print("failed:", img_file)
