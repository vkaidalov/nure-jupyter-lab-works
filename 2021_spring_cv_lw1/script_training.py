import random
import sys

import cv2
import joblib
import numpy as np
import skimage.feature
import sklearn.svm


TRUE_LOCATIONS_FILENAME = 'train_processed.idl'
TRAIN_IMG_DIR = 'train'
PATCH_HEIGHT = 200
PATCH_WIDTH = 80
PEDESTRIAN_LABEL = 1
BACKGROUND_LABEL = 2
RESULTS_TABLE_ROWS = 4
RESULTS_TABLE_COLS = 10
RESULTS_TABLE_COUNT = RESULTS_TABLE_ROWS * RESULTS_TABLE_COLS
SVM_MODEL_FILENAME = 'trained_svm_model.joblib'

with open('data/train-processed.idl') as f:
    lines = f.readlines()

lines = [s for s in lines if s != '']

patches = []
labels = []  # 1 for pedestrians, 2 for backgrounds

for line in lines:
    img_id, y0, x0, y1, x1 = [int(x) for x in line.split()]

    img_filepath = f'data/{TRAIN_IMG_DIR}/{img_id}.png'
    img = cv2.imread(img_filepath)

    img_pedestrian = img[y0:y1, x0:x1]
    patches.append(img_pedestrian)
    labels.append(PEDESTRIAN_LABEL)

    height, width, _channels = img.shape

    current_x = 0
    while current_x + PATCH_WIDTH <= x0:
        patches.append(
            img[0:PATCH_HEIGHT, current_x:(current_x + PATCH_WIDTH)]
        )
        labels.append(BACKGROUND_LABEL)
        current_x += PATCH_WIDTH

    current_x = x1
    while current_x + PATCH_WIDTH <= width:
        patches.append(
            img[0:PATCH_HEIGHT, current_x:(current_x + PATCH_WIDTH)]
        )
        labels.append(BACKGROUND_LABEL)
        current_x += PATCH_WIDTH

print(f'Pedestrian patches have been loaded: {labels.count(PEDESTRIAN_LABEL)}')
print(f'Background patches have been loaded: {labels.count(BACKGROUND_LABEL)}')

patches_gray = [
    cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) for img in patches
]

hog_images = []
hog_features = []
print('Started calculating HOGs...')
for patch_gray in patches_gray:
    features, hog_image = skimage.feature.hog(
        patch_gray,
        orientations=8,
        pixels_per_cell=(16, 16),
        cells_per_block=(4, 4),
        block_norm='L2',
        visualize=True
    )
    hog_features.append(features)
    hog_images.append(hog_image)
print('Finished calculating HOGs.')

if '--show-random-results' in sys.argv:
    # The lengths of these lists must be bigger than RESULTS_TABLE_COUNT.
    pedestrian_indexes = [i for (i, v) in enumerate(labels) if v == PEDESTRIAN_LABEL]
    background_indexes = [i for (i, v) in enumerate(labels) if v == BACKGROUND_LABEL]

    random.shuffle(pedestrian_indexes)
    random.shuffle(background_indexes)

    pedestrian_indexes = pedestrian_indexes[:RESULTS_TABLE_COUNT]
    background_indexes = background_indexes[:RESULTS_TABLE_COUNT]

    def build_table_img(indexes, images):
        table_img = None
        for i in range(RESULTS_TABLE_ROWS):
            row_img = None
            for j in range(RESULTS_TABLE_COLS):
                idx = i * RESULTS_TABLE_COLS + j
                patch = images[indexes[idx]]
                row_img = np.hstack((row_img, patch)) if j else patch
            table_img = np.vstack((table_img, row_img)) if i else row_img
        return table_img

    cv2.imshow('Random Pedestrians', build_table_img(pedestrian_indexes, patches))
    cv2.imshow('HOGs of Random Pedestrians', build_table_img(pedestrian_indexes, hog_images))
    cv2.imshow('Random Backgrounds', build_table_img(background_indexes, patches))
    cv2.imshow('HOGs of Random Backgrounds', build_table_img(background_indexes, hog_images))
    cv2.waitKey(0)

svm_classifier = sklearn.svm.SVC()

print('Starting training the SVM classifier...')
svm_classifier.fit(hog_features, labels)
print('Finished training.')

print(f'Saving the trained SVM classifier to {SVM_MODEL_FILENAME}...')
joblib.dump(svm_classifier, f'data/{SVM_MODEL_FILENAME}')
print('Finished saving.')
