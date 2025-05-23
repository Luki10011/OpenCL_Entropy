import os
import cv2
from config import *


def main():

    for filename in os.listdir(IMG_DIR):
        if filename.endswith(".tif"):
            filepath = os.path.join(IMG_DIR, filename)
            img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

            output_path = os.path.join(IMG_DIR, filename.split('.')[0] + ".bmp")
            cv2.imwrite(output_path, img)


if __name__ == "__main__":
    main()