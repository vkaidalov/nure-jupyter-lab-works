import os
from typing import Dict, List


TRUE_TEST_LOCATIONS_FILENAME = 'test-processed.idl'
PREDICTIONS_FILENAME = 'predictions.idl'
PATCH_WIDTH = 80

true_x_locations_dict: Dict[int, int] = {}

with open(os.path.join('data', TRUE_TEST_LOCATIONS_FILENAME)) as f:
    lines = f.readlines()
lines = [line for line in lines if line != '']

for line in lines:
    img_id, y0, x0, y1, x1 = [int(x) for x in line.split()]
    true_x_locations_dict[img_id] = x0

predicted_x_locations_dict: Dict[int, List[int]] = {}

with open(os.path.join('data', PREDICTIONS_FILENAME)) as f:
    lines = f.readlines()
lines = [line for line in lines if line != '']

for line in lines:
    img_id, y0, x0, y1, x1 = [int(x) for x in line.split()]
    predicted_x_locations_dict[img_id] = predicted_x_locations_dict.get(img_id, []) + [x0]

true_detected = 0
true_positives = 0
false_positives = 0
false_negatives = 0

for img_id, true_x in true_x_locations_dict.items():
    current_predicted_x_locations = predicted_x_locations_dict.get(img_id, [])

    current_true_positives = len([
        x for x in current_predicted_x_locations
        if true_x - PATCH_WIDTH / 2 <= x <= true_x + PATCH_WIDTH / 2
    ])

    true_positives += current_true_positives
    false_positives += len(current_predicted_x_locations) - current_true_positives

    if current_true_positives > 0:
        true_detected += 1
    else:
        false_negatives += 1

print('IDs of the images that doesn\'t have any pedestrians:')
for img_id in predicted_x_locations_dict:
    if img_id not in true_x_locations_dict:
        print(img_id)
        false_positives += len(predicted_x_locations_dict[img_id])

print(f'True number of pedestrians: {len(true_x_locations_dict)}')
print(f'Detected number of pedestrians: {true_detected}')
print(f'Total number of predictions: {sum(len(l) for l in predicted_x_locations_dict.values())}')
print(f'True Positives (including overlapping): {true_positives}')
print(f'False Positives: {false_positives}')
print(f'False Negatives: {false_negatives}')
print(f'Recall: {round(true_detected / len(true_x_locations_dict), 2)}')
print(f'Precision: {round(true_positives / (true_positives + false_positives), 2)}')
