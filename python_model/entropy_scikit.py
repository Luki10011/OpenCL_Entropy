from skimage.filters.rank import entropy
from skimage.morphology import disk, footprint_rectangle
import cv2
import os
import numpy as np
import time

from config import *


def test_entropy(img_name: str, struct_elem: np.ndarray, struct_elem_name: str) -> float:
    img = cv2.imread(os.path.join(IMG_DIR, img_name), 0)
    
    start = time.time()
    entr_img = cv2.normalize(entropy(img, struct_elem), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    comp_time = time.time() - start
    print(f"Elapsed time for image {img_name}: {comp_time} s")
    
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
        
    cv2.imwrite(os.path.join(OUT_DIR, f"{img_name.split('.')[0]}_ModelOut_{struct_elem_name}.bmp"), entr_img)
    
    return comp_time


def main():
    # Uwaga !!! parameter disk to promień i to bez środka!
    struct_elem = disk(2)
    print(struct_elem)
    
    for filename in os.listdir(IMG_DIR):
        if not filename.endswith(".tif"):
            test_entropy(filename, struct_elem, "circle5x5")

    struct_elem = footprint_rectangle((5,5))
    print(struct_elem)
    
    for filename in os.listdir(IMG_DIR):
        if not filename.endswith(".tif"):
            test_entropy(filename, struct_elem, "square5x5")
    
if __name__ == "__main__":
    main()
