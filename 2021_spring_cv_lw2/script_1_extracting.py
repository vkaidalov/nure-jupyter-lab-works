import os
from typing import Dict

import cv2
import numpy as np

import constants


with open(constants.SIMPLE_BOXES_FILEPATH) as f:
    lines = f.readlines()
lines = [line for line in lines if line != '']

objects_num = len(lines)
object_per_image_counter: Dict[int, int] = {}
width_list = []
height_list = []

os.makedirs(constants.SIMPLE_PATCHES_PATH, exist_ok=True)

for line in lines:
    img_id, y0, x0, y1, x1 = [int(x) for x in line.split()]
    curr_width = x1 - x0
    curr_height = y1 - y0
    width_list.append(curr_width)
    height_list.append(curr_height)
    object_per_image_counter[img_id] = object_per_image_counter.get(img_id, 0) + 1
    patch_num = object_per_image_counter[img_id]
    patch_filepath = os.path.join(
        constants.SIMPLE_PATCHES_PATH,
        f'{img_id}_{patch_num}.jpg'
    )

    source_filename = f'{img_id}.jpg'

    source_img = cv2.imread(
        os.path.join(constants.SIMPLE_IMAGES_PATH, source_filename)
    )
    source_img_h, source_img_w, _ = source_img.shape

    print(f'{source_filename}\t{source_img_h}\t*\t{source_img_w}')
    print(f'{patch_num}\t{curr_height}\t*\t{curr_width}')
    print(f'Saving patch #{patch_num} of {source_filename}...')

    blank_patch = np.zeros(
        (constants.UNIFIED_PATCH_HEIGHT, constants.UNIFIED_PATCH_WIDTH, 3),
        np.uint8
    )
    y_offset = round((constants.UNIFIED_PATCH_HEIGHT - curr_height) / 2)
    x_offset = round((constants.UNIFIED_PATCH_WIDTH - curr_width) / 2)
    current_patch = source_img[y0:y1, x0:x1]
    blank_patch[y_offset:(y_offset + curr_height), x_offset:(x_offset + curr_width)] = current_patch

    cv2.imwrite(patch_filepath, blank_patch)

print(f'Total number of the objects: {objects_num}')
print(f'AVG width: {sum(width_list) / objects_num}')
print(f'MAX width: {max(width_list)}')
print(f'MIN width: {min(width_list)}')
print(f'AVG height: {sum(height_list) / objects_num}')
print(f'MAX height: {max(height_list)}')
print(f'MIN height: {min(height_list)}')
