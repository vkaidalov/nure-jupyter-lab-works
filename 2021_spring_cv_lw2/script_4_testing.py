import sys
import joblib

import cv2
import skimage.feature

import constants


DX_STEP = 10
DY_STEP = 10

WINDOW_MIN_HEIGHT = 97
WINDOW_MAX_HEIGHT = 97
WINDOW_HEIGHT_STEP = 10
WINDOW_MIN_WIDTH = 97
WINDOW_MAX_WIDTH = 97
WINDOW_WIDTH_STEP = 10
svm_classifier = joblib.load(constants.SVM_MODEL_FILEPATH)

input_img_filepath = sys.argv[1]

delay_ms = int(sys.argv[2]) if len(sys.argv) > 2 else 0
print(delay_ms)

input_img = cv2.imread(input_img_filepath)
height, width, _channels = input_img.shape

predictions = []

for curr_win_h in range(WINDOW_MIN_HEIGHT, WINDOW_MAX_HEIGHT + 1, WINDOW_HEIGHT_STEP):
    for curr_win_w in range(WINDOW_MIN_WIDTH, WINDOW_MAX_WIDTH + 1, WINDOW_WIDTH_STEP):
        for curr_x in range(0, width - curr_win_w, DX_STEP):
            for curr_y in range(0, height - curr_win_h, DY_STEP):
                curr_patch = input_img[curr_y:(curr_y + curr_win_h), curr_x:(curr_x + curr_win_w)]
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
                    predictions.append(
                        (curr_x, curr_y, curr_x + curr_win_w, curr_y + curr_win_h)
                    )
                    print(predictions[-1])

for prediction in predictions:
    input_img = cv2.rectangle(
        input_img, (prediction[0], prediction[1]), (prediction[2], prediction[3]),
        (0, 255, 0), 2
    )

cv2.imshow(input_img_filepath, input_img)
cv2.waitKey(delay_ms)
