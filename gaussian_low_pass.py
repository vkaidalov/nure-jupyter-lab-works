import math

import numpy as np


def distance(point1, point2):
    return math.sqrt(
        (point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2
    )


def gaussian_low_pass(D0, imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows / 2, cols / 2)
    for x in range(cols):
        for y in range(rows):
            base[y, x] = math.exp(
                ((-distance((y, x), center) ** 2) / (2 * (D0 ** 2)))
            )
    return base
