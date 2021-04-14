from collections import namedtuple, defaultdict
from typing import List, Tuple, DefaultDict

import constants


Point = namedtuple('Point', ['x', 'y'])


def get_default_dict_of_rects(filepath) -> DefaultDict[int, List[Tuple[Point, Point]]]:
    result = defaultdict(list)

    with open(filepath) as f:
        lines = f.readlines()
    lines = [line for line in lines if line != '']

    for line in lines:
        img_id, y0, x0, y1, x1 = [int(x) for x in line.split()]
        result[img_id].append(
            (Point(x0, y0), Point(x1, y1))
        )

    return result


def intersection_area(a: Tuple[Point, Point], b: Tuple[Point, Point]):
    dx = min(a[1].x, b[1].x) - max(a[0].x, b[0].x)
    dy = min(a[1].y, b[1].y) - max(a[0].y, b[0].y)
    if (dx >= 0) and (dy >= 0):
        return dx * dy
    else:
        return 0


def area(a: Tuple[Point, Point]):
    return (a[1].y - a[0].y) * (a[1].x - a[0].x)


true_boxes = get_default_dict_of_rects(constants.TEST_BOXES_FILEPATH)
predictions = get_default_dict_of_rects(constants.PREDICTIONS_FILEPATH)

true_people_num = sum(len(l) for l in true_boxes.values())
predictions_num = sum(len(l) for l in predictions.values())

true_detected = 0
true_positives = 0
false_positives = 0
false_negatives = 0

for k in predictions:
    if k not in true_boxes:
        false_positives += len(predictions[k])

for k in true_boxes:
    curr_predictions = predictions.get(k, [])[:]  # create a copy

    for true_box in true_boxes[k]:
        true_box_area = area(true_box)
        to_leave = []
        detected = False
        for curr_prediction in curr_predictions:
            curr_prediction_area = area(curr_prediction)
            inter_area = intersection_area(true_box, curr_prediction)
            overlapping = (
                inter_area / (true_box_area + curr_prediction_area - inter_area)
            )
            if overlapping < 0.5:
                to_leave.append(curr_prediction)
            else:
                true_positives += 1
                detected = True
        if detected:
            true_detected += 1
        else:
            false_negatives += 1
        curr_predictions = to_leave

    false_positives += len(curr_predictions)

assert true_people_num == true_detected + false_negatives
assert predictions_num == true_positives + false_positives

print(f'True number of people: {true_people_num}')
print(f'Number of people detected: {true_detected}')
print(f'Total number of predictions: {predictions_num}')
print(f'True Positives (including overlapping): {true_positives}')
print(f'False Positives: {false_positives}')
print(f'False Negatives: {false_negatives}')
print(f'Recall: {round(true_detected / true_people_num, 2)}')
print(f'Precision: {round(true_positives / (true_positives + false_positives), 2)}')
