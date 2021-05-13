import sys
import os

import numpy as np
from PIL import Image

from utils import read_image, rgb


TEST_IMAGES_DIR = 'data/test_images'
PIXEL_BLACK = np.array([0, 0, 0])
PIXEL_WHITE = np.array([255, 255, 255])
PIXEL_RED = np.array([255, 0, 0])


def run(image_path):
    k = 0.725
    pixels_bytes, height, width = read_image(image_path)
    pixels_bw, image = get_pixels_array(pixels_bytes, height, width, k)
    cell_rows = get_cell_rows(pixels_bw, height, width)
    cell_rows = clear_cell_rows(cell_rows)
    img = draw_border(cell_rows, image)

    Image.open(image_path, 'r').show()
    Image.fromarray(img, 'RGB').show()
    Image.fromarray(pixels_bw, 'RGB').show()


def get_pixels_array(pixels_bytes, height, width, k):
    pixels = np.empty((height, width), dtype=np.uint8)
    image = np.empty((height, width, 3), dtype=np.uint8)
    pixels_bw = np.empty((height, width, 3), dtype=np.uint8)
    max_pixels = []

    i = 0
    for row_index in range(height):
        pixels_sum = 0
        for column_index in range(width):
            image[height - row_index - 1, column_index] = rgb(pixels_bytes[i])
            pixels[height - row_index - 1, column_index] = pixel_avg(rgb(pixels_bytes[i]))
            i += 1
        max_pixels.append(max(pixels[height - row_index - 1]))
        i -= width

        row_max = max_pixels[row_index]
        for column_index in range(width):
            pixel = pixel_avg(rgb(pixels_bytes[i]))

            value = PIXEL_BLACK
            if pixels_sum >= k * row_max:
                value = PIXEL_WHITE
                pixels_sum -= k * row_max
            pixels_sum += pixel

            pixels_bw[height - row_index - 1, column_index] = value
            i += 1

    return pixels_bw, image


def get_cell_rows(pixels_bw, height, width):
    prev = PIXEL_WHITE
    cell_rows = {}
    for row_index in range(height):
        switch, whites = 0, 0
        left_end, right_end = -1, -1
        left = True
        for column_index in range(width):
            pixel = pixels_bw[row_index, column_index]

            if not pixels_equal(prev, pixel):
                prev = pixels_bw[row_index, column_index]
                if left_end == -1:
                    left_end = column_index
                whites = 0 if pixels_equal(pixel, PIXEL_BLACK) else 1
                switch += 1

            if pixels_equal(pixel, PIXEL_WHITE):
                whites += 1

            if whites > 3:
                if not left:
                    right_end = column_index
                    cell_rows[row_index][1] = right_end
                    break
                switch, whites = 0, 0
                left_end = -1

            if switch > 2 and whites <= 3:
                if left:
                    left = False
                    cell_rows[row_index] = [left_end, width]
    return cell_rows


def clear_cell_rows(cell_rows):
    prev = min(cell_rows.keys())
    to_delete = []
    for row_index, ends in cell_rows.items():
        if row_index - prev > 1:
            to_delete.append(prev)
        prev = row_index

    for item in to_delete:
        del cell_rows[item]

    return cell_rows


def draw_border(figure_rows, image):
    upper_border = max(figure_rows.keys())
    lower_border = min(figure_rows.keys())
    left_border, right_border = np.inf, 0
    for row_index, ends in figure_rows.items():
        if ends[0] < left_border:
            left_border = ends[0]
        if ends[1] > right_border:
            right_border = ends[1]

    for column_index in range(left_border, right_border):
        image[upper_border, column_index] = PIXEL_RED
        image[lower_border, column_index] = PIXEL_RED

    for row_index in range(lower_border, upper_border):
        image[row_index, left_border - 1] = PIXEL_RED
        image[row_index, right_border - 1] = PIXEL_RED

    return image


def pixel_avg(pixel):
    return sum(pixel) // len(pixel)


def pixels_equal(pixel_first, pixel_second):
    return pixel_first.all() == pixel_second.all()


if __name__ == '__main__':
    if len(sys.argv) == 1:
        for filename, _ in zip(
                [name for name in os.listdir(TEST_IMAGES_DIR) if name.endswith('.bmp')],
                range(5)
        ):
            filepath = os.path.join(TEST_IMAGES_DIR, filename)
            print(filepath)
            run(filepath)
    else:
        input_img_filepath = sys.argv[1]
        run(input_img_filepath)
