from skimage.filters.rank import entropy
from skimage.morphology import disk
import cv2
import os
import numpy as np
import time


IMG_DIR = "test_images"
OUT_DIR = "out"


def test_entropy(img_name: str, struct_elem: np.ndarray) -> float:
    img = cv2.imread(os.path.join(IMG_DIR, img_name), 0)
    
    start = time.time()
    entr_img = cv2.normalize(entropy(img, struct_elem), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    comp_time = time.time() - start
    print(f"Elapsed time for image {img_name}: {comp_time} s")
    
    if not os.path.exists(OUT_DIR):
        os.mkdir(OUT_DIR)
        
    cv2.imwrite(os.path.join(OUT_DIR, "entr_img.png"), entr_img)
    
    return comp_time


def main():
    struct_elem = disk(5)
    
    test_entropy("cameraman.png", struct_elem)
    test_entropy("scikit_cameraman.png", struct_elem)
    test_entropy("circuit.png", struct_elem)
    test_entropy("peppers.png", struct_elem)
    
if __name__ == "__main__":
    main()
