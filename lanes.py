import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import ImageGrab
import time


def make_coordinates(image, line_parameters):
    try:
        slope, intercept = line_parameters
        y1 = image.shape[0]
        y2 = int(y1 * (4 / 8))
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])
    except Exception as e:
        print(str(e))


def canny(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)             # reduce noise in the grayscale image
    canny = cv2.Canny(blur, 50, 150)
    return canny


def average_slope_intercept(image, lines):
    left_fit = []
    right_fit = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]                             # lines on the left will have a negative slope, lines on the right will have a positive slope
            intercept = parameters[1]
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    left_fit_average = np.average(left_fit, axis=0)
    right_fit_average = np.average(right_fit, axis=0)
    left_line = make_coordinates(image, left_fit_average)
    right_line = make_coordinates(image, right_fit_average)
    return np.array([left_line, right_line])


def region_of_interest(image):
    height, width = image.shape[0], image.shape[1]
    polygons = np.array([[(0, 800), (width, 800), (1000, 400)]])
    mask = np.zeros_like(image)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(image, mask)
    return masked_image


def display_lines(image, lines):
    line_image = np.zeros_like(image)
    try:
        if lines is not None:
            for x1, y1, x2, y2 in lines:
                cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 10)
        return line_image
    except Exception as e:
        print(str(e))


for i in list(range(4))[::-1]:
    print(i + 1)
    time.sleep(1)


last_time = time.time()
cap = cv2.VideoCapture("Van_driving.mp4")
while(cap.isOpened()):
    _, frame = cap.read()
    print('Frame took {} seconds'.format(time.time() - last_time))
    last_time = time.time()
    canny_image = canny(frame)
    cropped_image = region_of_interest(canny_image)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=5)      # size of the bins, degree precision, threshold: min num votes required to accept a candidate line, empty array, minLineLength, maxLineGap.
    averaged_lines = average_slope_intercept(frame, lines)
    line_image = display_lines(frame, averaged_lines)
    try:
        combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
    except Exception as e:
        print(str(e))

    cv2.imshow('result', combo_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# plt.imshow(lane_image)
# plt.show()
# polygons = np.array([[(200, height), (1100, height), (550, 250)]])
