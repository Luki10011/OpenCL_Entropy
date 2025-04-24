from skimage.filters.rank import entropy
from skimage.morphology import disk
import cv2
import os
import numpy as np


def main():
    images_dir = "test_images"
    out_dir = "out"
    struct_elem = disk(5)
    
    cameraman = cv2.imread(os.path.join(images_dir, "cameraman.png"), 0)
    scikit_cameraman = cv2.imread(os.path.join(images_dir, "scikit_cameraman.png"), 0)
    circuit = cv2.imread(os.path.join(images_dir, "circuit.png"), 0)
    peppers = cv2.imread(os.path.join(images_dir, "peppers.png"), 0)
    
    entr_cameraman = cv2.normalize(entropy(cameraman, struct_elem), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    entr_scikit_cameraman = cv2.normalize(entropy(scikit_cameraman, struct_elem), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    entr_circuit = cv2.normalize(entropy(circuit, struct_elem), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    entr_peppers = cv2.normalize(entropy(peppers, struct_elem), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
    
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        
    cv2.imwrite(os.path.join(out_dir, "entr_cameraman.png"), entr_cameraman)
    cv2.imwrite(os.path.join(out_dir, "entr_scikit_cameraman.png"), entr_scikit_cameraman)
    cv2.imwrite(os.path.join(out_dir, "entr_circuit.png"), entr_circuit)
    cv2.imwrite(os.path.join(out_dir, "entr_peppers.png"), entr_peppers)
    
    
if __name__ == "__main__":
    main()
