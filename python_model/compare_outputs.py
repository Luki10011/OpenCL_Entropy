from config import *
import os
import cv2
import numpy as np


def main():
    for model_out_filename in os.listdir(OUT_DIR):
        for cl_out_filename in os.listdir(CL_OUT_DIR):
            if "Our" in model_out_filename:  # warunek, żeby porównywało tylko z wynikami naszego modelu
                cl_out_name = cl_out_filename.split('_')[0]
                cl_out_struct = cl_out_filename.split('_')[-1].split('.')[0]
                model_out_name = model_out_filename.split('_')[0]
                model_out_struct = model_out_filename.split('_')[-1].split('.')[0]

                if model_out_name == cl_out_name and model_out_struct == cl_out_struct:
                    model_out_path = os.path.join(OUT_DIR, model_out_filename)
                    cl_out_path = os.path.join(CL_OUT_DIR, cl_out_filename)
                    model_img = cv2.imread(model_out_path, cv2.IMREAD_GRAYSCALE)
                    cl_img = cv2.imread(cl_out_path, cv2.IMREAD_GRAYSCALE)

                    diff_img = cv2.absdiff(model_img, cl_img)
                    # cv2.imshow("diff", diff_img)
                    # cv2.waitKey(0)
                    diff = np.sum(diff_img)

                    print(f"Sum of absolute difference for images {model_out_filename} and {cl_out_filename}: {diff}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()