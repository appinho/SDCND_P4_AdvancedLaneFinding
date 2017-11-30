import calibration
import line_detection
import os
import matplotlib.image as mpimg

# 1 CAMERA CALIBRATION
cal_mtx,dist_coeff = calibration.calibrate()
print(cal_mtx)
print(dist_coeff)
# 2 DISTORTION CORRECTION
line_detector = line_detection.LineDetector(cal_mtx,dist_coeff)
for image_name in os.listdir('test_images/'):
    print(image_name)
    image = mpimg.imread('test_images/' + image_name)
    line_detector.work_on_test_image(image)
# line_detector.adjust_parameters()
# 3 COLOR & GRADIENT THRESHOLD

# 4 PERSPECTIVE TRANSFORM

# 5 DETECT LINES

# 6 CALCULATE CURVATURE AND OFFSET

# 7 PLOT RESULTS