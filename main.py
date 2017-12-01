import calibration
import line_detection
import os
import cv2

# 1 CAMERA CALIBRATION
cal_mtx,dist_coeff = calibration.calibrate()
print(cal_mtx)
print(dist_coeff)
# 2 DISTORTION CORRECTION
line_detector = line_detection.LineDetector(cal_mtx,dist_coeff)
# for image_name in os.listdir('false_estimations/'):
#     print(image_name)
#     image = cv2.imread('false_estimations/' + image_name)
#     line_detector.process_image(image)
#     # line_detector.adjust_parameters(image)
# Test video
cap = cv2.VideoCapture('project_video.mp4')
# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'MP4V')
out = cv2.VideoWriter('output4.mp4',fourcc, 20.0, (1280,720))

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret==True:
        # write the flipped frame
        result = line_detector.process_image(frame)
        result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        out.write(result)

        # cv2.imshow('frame',result)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything if job is finished
cap.release()
out.release()
cv2.destroyAllWindows()