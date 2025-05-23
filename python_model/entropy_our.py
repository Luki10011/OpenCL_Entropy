import numpy as np
from math import log2
from PIL import Image
from config import *
import os
from skimage.morphology import disk, footprint_rectangle


def to_grayscale(input_image: np.ndarray) -> np.ndarray:
    """
    Konwertuje obraz RGB (uint8) do odcieni szarości (uint8).
    Zgodne z kernelem OpenCL.
    """
    if input_image.shape[2] == 4:
        input_image = input_image[:, :, :3]  # ignoruj kanał alpha, jeśli jest

    # Przekształcenie do szarości: 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray = (0.2989 * input_image[:, :, 0] +
            0.5870 * input_image[:, :, 1] +
            0.1140 * input_image[:, :, 2]).astype(np.uint8)

    return gray


def entropy_image(gray_image: np.ndarray, struct_elem: np.ndarray) -> np.ndarray:
    """
    Oblicza lokalną entropię obrazu w odcieniach szarości.
    """
    height, width = gray_image.shape
    m, n = struct_elem.shape
    pad_y, pad_x = m // 2, n // 2

    # Wypełnienie obrazu zerami
    padded_img = np.pad(gray_image, ((pad_y, pad_y), (pad_x, pad_x)), mode='constant', constant_values=0)
    output = np.zeros_like(gray_image, dtype=np.uint8)

    for y in range(height):
        for x in range(width):
            # Wycinanie okna i stosowanie maski
            window = padded_img[y:y + m, x:x + n]
            masked = window[struct_elem == 1].flatten()

            if masked.size == 0:
                entropy_val = 0.0
            else:
                # Histogram
                hist = np.bincount(masked, minlength=256)
                probs = hist[hist > 0] / masked.size
                entropy_val = -np.sum(probs * np.log2(probs))

                # Normalizacja
                max_entropy = log2(min(masked.size, 256))
                normalized_entropy = entropy_val / max_entropy * 255.0
                output[y, x] = np.uint8(np.clip(normalized_entropy, 0, 255))

    return output


def rgba_from_gray(gray_image: np.ndarray) -> np.ndarray:
    """
    Konwertuje obraz z szarości do RGBA (taki format ma OpenCLowy `uint4`).
    """
    h, w = gray_image.shape
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[:, :, 0] = gray_image
    rgba[:, :, 1] = gray_image
    rgba[:, :, 2] = gray_image
    rgba[:, :, 3] = 255  # Alpha
    return rgba

if __name__ == "__main__":
    struct_elems = disk(2), footprint_rectangle((5,5))
    struct_elem_names = "circle5x5", "square5x5"

    for filename in os.listdir(IMG_DIR):
        if not filename.endswith(".tif"):
            for struct_elem, struct_elem_name in zip(struct_elems, struct_elem_names):
                image = Image.open(os.path.join(IMG_DIR, filename)).convert("RGBA")
                image_np = np.array(image)

                gray = to_grayscale(image_np)

                entropy_result = entropy_image(gray, struct_elem)

                # Konwersja na RGBA do porównania z OpenCL
                entropy_rgba = rgba_from_gray(entropy_result)

                Image.fromarray(entropy_rgba).save(os.path.join(OUT_DIR, f"{filename.split('.')[0]}_OurModelOut_{struct_elem_name}.bmp"))
