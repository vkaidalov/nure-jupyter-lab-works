import os
from typing import Dict, List

import cv2
import joblib
import skimage.feature


STEP_SIZE = 20
PATCH_HEIGHT = 200
PATCH_WIDTH = 80
TEST_IMG_DIR = 'test-public'
SVM_MODEL_FILENAME = 'trained_svm_model.joblib'
PREDICTIONS_FILENAME = 'predictions.idl'
PEDESTRIAN_LABEL = 1
BACKGROUND_LABEL = 2

svm_classifier = joblib.load(f'data/{SVM_MODEL_FILENAME}')

predictions: Dict[int, List[List[int]]] = {}
for filename in os.listdir(os.path.join('data', TEST_IMG_DIR)):
    img_id = int(filename.split('.')[0])
    test_img = cv2.imread(os.path.join('data', TEST_IMG_DIR, filename))
    height, width, _channels = test_img.shape

    current_x = 0
    while current_x + PATCH_WIDTH <= width:
        x0, y0, x1, y1 = current_x, 0, current_x + PATCH_WIDTH, PATCH_HEIGHT
        current_patch = test_img[0:PATCH_HEIGHT, current_x:(current_x + PATCH_WIDTH)]
        current_patch = cv2.cvtColor(current_patch, cv2.COLOR_RGB2GRAY)
        features, _hog_image = skimage.feature.hog(
            current_patch,
            orientations=8,
            pixels_per_cell=(16, 16),
            cells_per_block=(4, 4),
            block_norm='L2',
            visualize=True
        )
        predicted_label = svm_classifier.predict([features])[0]
        if predicted_label == PEDESTRIAN_LABEL:
            predictions[img_id] = predictions.get(img_id, []) + [[y0, x0, y1, x1]]
        current_x += STEP_SIZE

with open(os.path.join('data', PREDICTIONS_FILENAME), 'w') as f:
    for img_id in sorted(predictions):
        for patch_coords in predictions[img_id]:
            f.write('\t'.join(str(i) for i in [img_id, *patch_coords]) + '\n')
