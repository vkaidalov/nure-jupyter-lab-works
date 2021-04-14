import os

import cv2
import joblib
import skimage.feature
import sklearn.svm

import constants


labels = []
patches = []

for filename in os.listdir(constants.SIMPLE_PATCHES_PATH):
    patches.append(
        cv2.imread(os.path.join(constants.SIMPLE_PATCHES_PATH, filename))
    )
    labels.append(constants.RIGHT_PATCH_LABEL)

for filename in os.listdir(constants.SIMPLE_WRONG_PATCHES_PATH):
    patches.append(
        cv2.imread(os.path.join(constants.SIMPLE_WRONG_PATCHES_PATH, filename))
    )
    labels.append(constants.WRONG_PATCH_LABEL)

print(f'RIGHT patches loaded: {labels.count(constants.RIGHT_PATCH_LABEL)}')
print(f'WRONG patches loaded: {labels.count(constants.WRONG_PATCH_LABEL)}')

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

svm_classifier = sklearn.svm.SVC()

print('Starting training the SVM classifier...')
svm_classifier.fit(hog_features, labels)
print('Finished training.')

print(f'Saving the trained SVM classifier to {constants.SVM_MODEL_FILENAME}...')
joblib.dump(svm_classifier, constants.SVM_MODEL_FILEPATH)
print('Finished saving.')
