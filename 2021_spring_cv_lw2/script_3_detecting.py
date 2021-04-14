import os
from collections import namedtuple
from typing import Dict, List, Tuple

import cv2
import joblib
import skimage.feature

import constants


Point = namedtuple('Point', ['x', 'y'])

DX_STEP = 10
DY_STEP = 10

WINDOW_MIN_HEIGHT = 97
WINDOW_MAX_HEIGHT = 97
WINDOW_HEIGHT_STEP = 10
WINDOW_MIN_WIDTH = 97
WINDOW_MAX_WIDTH = 97
WINDOW_WIDTH_STEP = 10

svm_classifier = joblib.load(constants.SVM_MODEL_FILEPATH)

predictions: Dict[int, List[Tuple[Point, Point]]] = {}
for filename in sorted(os.listdir(constants.TEST_IMAGES_PATH)):
    print(f'Processing {filename}...')
    img_id = int(filename.split('.')[0])
    test_img = cv2.cvtColor(
        cv2.imread(os.path.join(constants.TEST_IMAGES_PATH, filename)),
        cv2.COLOR_RGB2GRAY
    )
    height, width = test_img.shape

    curr_x, curr_y = 0, 0
    curr_win_w = WINDOW_MIN_WIDTH
    curr_win_h = WINDOW_MIN_HEIGHT
    for curr_win_h in range(WINDOW_MIN_HEIGHT, WINDOW_MAX_HEIGHT + 1, WINDOW_HEIGHT_STEP):
        for curr_win_w in range(WINDOW_MIN_WIDTH, WINDOW_MAX_WIDTH + 1, WINDOW_WIDTH_STEP):
            for curr_x in range(0, width - curr_win_w, DX_STEP):
                for curr_y in range(0, height - curr_win_h, DY_STEP):
                    curr_patch = test_img[curr_y:(curr_y + curr_win_h), curr_x:(curr_x + curr_win_w)]
                    features, _hog_image = skimage.feature.hog(
                        curr_patch,
                        orientations=8,
                        pixels_per_cell=(16, 16),
                        cells_per_block=(4, 4),
                        block_norm='L2',
                        visualize=True
                    )
                    predicted_label = svm_classifier.predict([features])[0]
                    if predicted_label == constants.RIGHT_PATCH_LABEL:
                        predictions[img_id] = predictions.get(img_id, []) + [
                            (Point(curr_x, curr_y), Point(curr_x + curr_win_w, curr_y + curr_win_h))
                        ]
                        print(predictions[img_id][-1])

with open(constants.PREDICTIONS_FILEPATH, 'w') as f:
    for img_id in sorted(predictions):
        for patch_coords in predictions[img_id]:
            f.write(
                ' '.join(
                    str(i) for i in [
                        img_id,
                        patch_coords[0].y, patch_coords[0].x,
                        patch_coords[1].y, patch_coords[1].x
                    ]
                ) + '\n'
            )
