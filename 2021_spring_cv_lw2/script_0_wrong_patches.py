import os
import random

import cv2

import constants


WRONG_PATCHES_PER_FILE = 5

os.makedirs(constants.SIMPLE_WRONG_PATCHES_PATH, exist_ok=True)

for filename in os.listdir(constants.SIMPLE_IMAGES_PATH):
    if not filename.endswith('.jpg'):
        continue

    filename_without_ext = filename[:-4]

    filepath = os.path.join(constants.SIMPLE_IMAGES_PATH, filename)
    img = cv2.imread(filepath)
    height, width, _ = img.shape

    for i in range(WRONG_PATCHES_PER_FILE):
        x0 = random.randrange(0, width - constants.UNIFIED_PATCH_WIDTH)
        y0 = random.randrange(0, height - constants.UNIFIED_PATCH_HEIGHT)
        curr_wrong_patch = img[
            y0:(y0 + constants.UNIFIED_PATCH_HEIGHT),
            x0:(x0 + constants.UNIFIED_PATCH_WIDTH)
        ]
        out_filepath = os.path.join(
            constants.SIMPLE_WRONG_PATCHES_PATH,
            f'{filename_without_ext}_{i}.jpg'
        )
        cv2.imwrite(out_filepath, curr_wrong_patch)
