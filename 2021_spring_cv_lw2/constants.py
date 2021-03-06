import os


DATA_DIR = 'data'

SIMPLE_BOXES_FILENAME = 'simple_boxes.txt'
SIMPLE_BOXES_FILEPATH = os.path.join(DATA_DIR, SIMPLE_BOXES_FILENAME)

SIMPLE_IMAGES_DIR = 'simple_images'
SIMPLE_IMAGES_PATH = os.path.join(DATA_DIR, SIMPLE_IMAGES_DIR)

TEST_BOXES_FILENAME = 'test_boxes.txt'
TEST_BOXES_FILEPATH = os.path.join(DATA_DIR, TEST_BOXES_FILENAME)

TEST_IMAGES_DIR = 'test_images'
TEST_IMAGES_PATH = os.path.join(DATA_DIR, TEST_IMAGES_DIR)

SIMPLE_PATCHES_DIR = 'simple_patches'
SIMPLE_PATCHES_PATH = os.path.join(DATA_DIR, SIMPLE_PATCHES_DIR)

SIMPLE_WRONG_PATCHES_DIR = 'simple_wrong_patches'
SIMPLE_WRONG_PATCHES_PATH = os.path.join(DATA_DIR, SIMPLE_WRONG_PATCHES_DIR)

PREDICTIONS_FILENAME = 'predictions.txt'
PREDICTIONS_FILEPATH = os.path.join(DATA_DIR, PREDICTIONS_FILENAME)

UNIFIED_PATCH_HEIGHT = 97
UNIFIED_PATCH_WIDTH = 97

RIGHT_PATCH_LABEL = 1
WRONG_PATCH_LABEL = 2

SVM_MODEL_FILENAME = 'trained_svm_model.joblib'
SVM_MODEL_FILEPATH = os.path.join(DATA_DIR, SVM_MODEL_FILENAME)
