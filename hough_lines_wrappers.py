import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_hough_lines(
    img_rgb,
    *,
    canny_min_threshold=100,
    canny_max_threshold=200,
    hough_rho_resolution=1,
    hough_theta_resolution=np.pi / 180,
    hough_vote_threshold=150,
    line_color=(0, 255, 0),
    line_width=2
):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_edges = cv2.Canny(img_gray, canny_min_threshold, canny_max_threshold)
    
    lines = cv2.HoughLines(
        img_edges,
        hough_rho_resolution, hough_theta_resolution, hough_vote_threshold
    )
    
    img_with_lines = img_rgb.copy()
    for lines_list in lines:
        for rho, theta in lines_list:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 10000 * (-b))
            y1 = int(y0 + 10000 * (a))
            x2 = int(x0 - 10000 * (-b))
            y2 = int(y0 - 10000 * (a))

            cv2.line(img_with_lines, (x1, y1), (x2, y2), line_color, line_width)
    
    return img_gray, img_edges, img_with_lines


def get_hough_lines_prob(
    img_rgb,
    *,
    canny_min_threshold=100,
    canny_max_threshold=200,
    hough_rho_resolution=1,
    hough_theta_resolution=np.pi / 180,
    hough_vote_threshold=100,
    min_line_len=100,
    max_line_gap=10,
    line_color=(0, 255, 0),
    line_width=2
):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    img_edges = cv2.Canny(img_gray, canny_min_threshold, canny_max_threshold)
    
    lines = cv2.HoughLinesP(
        img_edges,
        rho=hough_rho_resolution,
        theta=hough_theta_resolution,
        threshold=hough_vote_threshold,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )
    
    img_with_lines = img_rgb.copy()
    for lines_list in lines:
        for x1, y1, x2, y2 in lines_list:
            cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    return img_gray, img_edges, img_with_lines
