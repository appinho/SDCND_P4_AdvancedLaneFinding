import calibration
import line_detection

# 1 CAMERA CALIBRATION
cal_mtx,dist_coeff = calibration.calibrate()
print(cal_mtx)
print(dist_coeff)
# 2 DISTORTION CORRECTION
line_detector = line_detection.LineDetector(cal_mtx,dist_coeff)
line_detector.work_on_test_image()
# 3 COLOR & GRADIENT THRESHOLD

# 4 PERSPECTIVE TRANSFORM

# 5 DETECT LINES

# 6 CALCULATE CURVATURE AND OFFSET

# 7 PLOT RESULTS