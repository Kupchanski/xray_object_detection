import cv2
import random
import numpy as np


def read_rgb(image_path: str) -> np.array:
    bgr = cv2.imread(image_path)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

def random_resize(image1: np.array, image2: np.array, smin=0.05):
    s = random.uniform(smin, 1)
    scale_factor = s**(1/3)
    width = int(image1.shape[1] * scale_factor)
    height = int(image1.shape[0] * scale_factor)
    dim = (width, height)
    resized_image1 = cv2.resize(image1, dsize=dim, interpolation=cv2.INTER_AREA)
    resized_image2 = cv2.resize(image2, dsize=dim, interpolation=cv2.INTER_AREA)
    return resized_image1, resized_image2
