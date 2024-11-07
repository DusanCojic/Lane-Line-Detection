import math
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimage
import numpy as np
from moviepy.editor import VideoFileClip

def region_of_interest(image):
    height = image.shape[0]
    width = image.shape[1]

    # points of a triangle
    vertices = [
        (0, height), # bottom-left
        (width / 2 + 25, height / 2 + 25), # around middle
        (width, height), # bottom-right
    ]

    vertices = np.array([vertices], np.int32)

    mask = np.zeros_like(image) # all zeros in shape of the image
    if len(image.shape) > 2: # coloring the whole mask to white
        channel_count = image.shape[2]
        mask_color = (255,) * channel_count
    else:
        mask_color = 255

    cv2.fillPoly(mask, vertices, mask_color) # creating a white triangle shape

    # whole mask will be black, except the triangle, which would be white
    # when perforimg bitwise and operation with image and mask,
    # image parts that overlap with black part of the mask will be black
    # and all parts that overlap with white part of the mask will be colors from the image
    masked_image = cv2.bitwise_and(image, mask)

    return masked_image

# converting an image to a grayscale
def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def gaussian_blur(image):
    return cv2.GaussianBlur(image, (5, 5), 0)

# applying Canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

# performing Hough Transform to get lines
def hough_transform(image):
    return cv2.HoughLinesP(
        image,
        rho=6, # distance resolution in radians
        theta=np.pi / 60, # angle resolution in radians
        threshold=160, # the minimum number of votes needed to detect a line
        minLineLength=40, # minimum length of a segment to be detected, in pixels
        maxLineGap=25 # maximum gap between segments to be considered as a single line
    )

def draw_lines(image, lines, color=[0, 0, 255], thickness=15):
    if (lines is None):
        return
    
    img = np.copy(image)  # creating a copy of an original image
    line_image = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)  # empty image with the dimensions of the original one

    # draw all lines on the line image
    for line in lines:
        for x1, y1, x2, y2 in line:
            # Ensure that x1, y1, x2, y2 are integers
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.line(line_image, (x1, y1), (x2, y2), color, thickness)

    # merge the line image with the original image
    img = cv2.addWeighted(img, 0.8, line_image, 1.0, 0.0)

    return img


def calculate_points(image, line):
    slope, intercept = line

    y1 = image.shape[0]
    y2 = int(y1 * 4 / 6)

    x1 = int((y1 - intercept) // slope)
    x2 = int((y2 - intercept) // slope)

    return [x1, y1, x2, y2]

def group_lines(image, lines):
    left_lines = []
    right_lines = []

    for line in lines:
        x1, y1, x2, y2 = line[0]

        if x1 == x2:  # avoid vertical lines
            continue

        parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Fit a line to the points

        slope = parameters[0]
        intercept = parameters[1]

        if slope < 0:
            left_lines.append((slope, intercept))
        else:
            right_lines.append((slope, intercept))

    if left_lines:
        # Compute the average of slopes and intercepts separately
        left_slopes, left_intercepts = zip(*left_lines)
        left_average = (np.mean(left_slopes), np.mean(left_intercepts))
        left_line = calculate_points(image, left_average)
    else:
        left_line = [0, 0, 0, 0]  # Empty if no lines found

    if right_lines:
        right_slopes, right_intercepts = zip(*right_lines)
        right_average = (np.mean(right_slopes), np.mean(right_intercepts))
        right_line = calculate_points(image, right_average)
    else:
        right_line = [0, 0, 0, 0]  # Empty if no lines found

    return [
        [left_line],
        [right_line],
    ]

def process_frame(image):
    gray_image = grayscale(image)
    smooth = gaussian_blur(gray_image)
    cannyed_image = canny(smooth)
    cropped_image = region_of_interest(cannyed_image)

    lines = hough_transform(cropped_image)
    result = draw_lines(image, group_lines(image, lines))

    return result

def process_video(video):
    input = VideoFileClip(video, audio=False)
    processed = input.fl_image(process_frame)
    processed.write_videofile("output.mp4", audio=False)

process_video("videos/1.mp4")