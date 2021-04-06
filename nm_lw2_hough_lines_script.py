"""
To use this script, please create a directory named 
INPUT_DIR (see the value below) and place an image file in it.
Create a PARAMS_FILENAME and fill it with values. Take the
'nw_lw2_script_input.example.txt' as an example.
Run the script.
As a result, two output image files will be created in the
OUTPUT_DIR directory:
- an image file showing results of the Standard Hough Line Transform
- an image file showing results of the Probabilistic Hough Line Transform
"""

import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from hough_lines_wrappers import get_hough_lines, get_hough_lines_prob


# Create it using nm_lw2_script_input.example.txt as an example
PARAMS_FILENAME = 'nm_lw2_script_input.txt'

with open(PARAMS_FILENAME) as file:
    lines = file.read().splitlines()

# Should exist within INP_IMG_DIR
INPUT_DIR = 'nm_lw2_hough_lines_input_images'
IMG_FILENAME = lines[0]
IMG_FILENAME_WO_EXT = '.'.join(IMG_FILENAME.split('.')[:-1])
INPUT_IMG_PATH = os.path.join(INPUT_DIR, IMG_FILENAME)
OUTPUT_DIR = 'nm_lw2_hough_lines_output_images'

canny_min_threshold_1 = int(lines[1])
canny_max_threshold_1 = int(lines[2])
hough_vote_threshold_1 = int(lines[3])

canny_min_threshold_2 = int(lines[4])
canny_max_threshold_2 = int(lines[5])
hough_vote_threshold_2 = int(lines[6])
min_line_len = int(lines[7])
max_line_gap = int(lines[8])

os.makedirs(OUTPUT_DIR, exist_ok=True)

img_rgb = cv2.cvtColor(cv2.imread(INPUT_IMG_PATH), cv2.COLOR_BGR2RGB)

# Hough Lines
img_gray, img_edges, img_with_lines = get_hough_lines(
    img_rgb,
    canny_min_threshold=canny_max_threshold_1,
    canny_max_threshold=canny_max_threshold_1,
    hough_vote_threshold=hough_vote_threshold_1
)

plt.figure(figsize=(6.4*4, 4.8*4), constrained_layout=False)

plt.subplot(221)
plt.imshow(img_rgb)
plt.title(IMG_FILENAME)

plt.subplot(222)
plt.imshow(img_gray, 'gray')
plt.title('Grayscale')

plt.subplot(223)
plt.imshow(img_edges, 'gray')
plt.title(f'Canny Edge Detection (min_thres={canny_min_threshold_1}, max_thres={canny_max_threshold_1})')

plt.subplot(224)
plt.imshow(img_with_lines)
plt.title(f'Hough Lines (min_votes={hough_vote_threshold_1})')

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        f'{IMG_FILENAME_WO_EXT}_hough_lines_{canny_min_threshold_1}_{canny_max_threshold_1}_{hough_vote_threshold_1}.png'
    ),
    bbox_inches='tight'
)

plt.close('all')

# Probabilistic Hough Lines
img_gray, img_edges, img_with_lines = get_hough_lines_prob(
    img_rgb,
    canny_min_threshold=canny_max_threshold_2,
    canny_max_threshold=canny_max_threshold_2,
    hough_vote_threshold=hough_vote_threshold_2,
    min_line_len=min_line_len,
    max_line_gap=max_line_gap
)

plt.figure(figsize=(6.4*4, 4.8*4), constrained_layout=False)

plt.subplot(221)
plt.imshow(img_rgb)
plt.title(IMG_FILENAME)

plt.subplot(222)
plt.imshow(img_gray, 'gray')
plt.title('Grayscale')

plt.subplot(223)
plt.imshow(img_edges, 'gray')
plt.title(f'Canny Edge Detection (min_thres={canny_min_threshold_2}, max_thres={canny_max_threshold_2})')

plt.subplot(224)
plt.imshow(img_with_lines)
plt.title(f'Hough Lines (min_votes={hough_vote_threshold_2}, min_len={min_line_len}, max_gap={max_line_gap})')

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        f'{IMG_FILENAME_WO_EXT}_hough_lines_prob_{canny_min_threshold_2}_{canny_max_threshold_2}_{hough_vote_threshold_2}_'
        f'{min_line_len}_{max_line_gap}.png'
    ),
    bbox_inches='tight'
)

plt.close('all')
