import sys
import joblib

import cv2
import skimage.feature


STEP_SIZE = 20
PEDESTRIAN_LABEL = 1
PATCH_HEIGHT = 200
PATCH_WIDTH = 80
SVM_MODEL_FILENAME = 'trained_svm_model.joblib'

svm_classifier = joblib.load(f'data/{SVM_MODEL_FILENAME}')

input_img_filepath = sys.argv[1]
input_img = cv2.imread(input_img_filepath)
height, width, _channels = input_img.shape

predictions = []

current_x = 0
while current_x + PATCH_WIDTH <= width:
    x0, y0, x1, y1 = current_x, 0, current_x + PATCH_WIDTH, PATCH_HEIGHT
    current_patch = input_img[0:PATCH_HEIGHT, current_x:(current_x + PATCH_WIDTH)]
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
        predictions.append([x0, y0, x1, y1])
    current_x += STEP_SIZE

for prediction in predictions:
    input_img = cv2.rectangle(
        input_img, (prediction[0], prediction[1]), (prediction[2], prediction[3]),
        (0, 255, 0), 2
    )

cv2.imshow(input_img_filepath, input_img)
cv2.waitKey(0)
