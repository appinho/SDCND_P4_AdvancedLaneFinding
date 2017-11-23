import matplotlib.image as mpimg
import cv2
import visualizer
import numpy as np

class LineDetector():
    def __init__(self,C, coeff):
        """
        Constructor
        :param C: Calibration matrix
        :param coeff: Distortion coefficients
        """
        self.C = C
        self.coeff = coeff

        self.thresh_min_sobel_x = 30
        self.thresh_max_sobel_x = 255
        self.thresh_min_sobel_y = 50
        self.thresh_max_sobel_y = 255
        self.thresh_min_grad_dir = 0.9
        self.thresh_max_grad_dir = 1.1
        self.thresh_min_grad_mag = 50
        self.thresh_max_grad_mag = 200
        self.kernel_size = 3

        self.debug = True

    def distortion_correction(self,image):
        """
        Distortion correction
        :param image: distorted image
        :return: undistorted image
        """
        return cv2.undistort(image, self.C, self.coeff, None, self.C)

    def create_thresholded_image(self,image):
        """
        Filters input image with sobel operator and gradients
        :param image: undistorted image
        :return: binary image
        """
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        # Apply each of the thresholding functions
        gradx = self.abs_sobel_thresh(gray, orient='x', sobel_kernel=self.kernel_size,
                                      thresh=(self.thresh_min_sobel_x, self.thresh_max_sobel_x))
        grady = self.abs_sobel_thresh(gray, orient='y', sobel_kernel=self.kernel_size,
                                      thresh=(self.thresh_min_sobel_y, self.thresh_max_sobel_y))
        mag_binary = self.mag_thresh(gray, sobel_kernel=self.kernel_size,
                                      thresh=(self.thresh_min_grad_mag, self.thresh_max_grad_mag))
        dir_binary = self.dir_threshold(gray, sobel_kernel=self.kernel_size,
                                      thresh=(self.thresh_min_grad_dir, self.thresh_max_grad_dir))

        combined = np.zeros_like(dir_binary)
        combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
        return combined

    def perspective_transform(self,image,src,dst):
        M = cv2.getPerspectiveTransform(src, dst)
        img_size = (image.shape[1], image.shape[0])
        warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
        return warped


    def abs_sobel_thresh(self, gray, orient='x', sobel_kernel=3, thresh=(0, 255)):
        # 2) Take the derivative in x or y given orient = 'x' or 'y'
        if orient == 'x':
            sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        else:
            sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the derivative or gradient
        abs_sobel = np.absolute(sobel)
        # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
        scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
        # 5) Create a mask of 1's where the scaled gradient magnitude
        grad_binary = np.zeros_like(scaled_sobel)
        grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return grad_binary

    def mag_thresh(self, gray, sobel_kernel=3, thresh=(0, 255)):
        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Calculate the magnitude
        magn = np.sqrt(sobelx * sobelx + sobely * sobely)
        # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
        scaled_magn = np.uint8(255 * magn / np.max(magn))
        # 5) Create a binary mask where mag thresholds are met
        mag_binary = np.zeros_like(scaled_magn)
        mag_binary[(scaled_magn >= thresh[0]) & (scaled_magn <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return mag_binary

    def dir_threshold(self, gray, sobel_kernel=3, thresh=(0, np.pi / 2)):

        # 2) Take the gradient in x and y separately
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
        # 3) Take the absolute value of the x and y gradients
        abs_sobelx = np.absolute(sobelx)
        abs_sobely = np.absolute(sobely)
        # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
        direction = np.arctan2(abs_sobely, abs_sobelx)
        # 5) Create a binary mask where direction thresholds are met
        dir_binary = np.zeros_like(direction)
        dir_binary[(direction >= thresh[0]) & (direction <= thresh[1])] = 1
        # 6) Return this mask as your binary_output image
        return dir_binary

    def work_on_test_image(self):
        """
        Test function to propagate a test image
        :return: None
        """

        # Read image
        image = mpimg.imread('test_images/test5.jpg')
        img_size = image.shape
        print(img_size)

        # Distortion correction
        dist = self.distortion_correction(image)
        # if self.debug:
        #     visualizer.plot_distortion_correction(image,dst)

        # Get thresholded binary image
        binary_image = self.create_thresholded_image(dist)
        # if self.debug:
        #     visualizer.plot_distortion_correction(image,binary_image)

        # Perspective transform
        src = np.float32(
            [[(img_size[1] / 2) - 100, img_size[0] / 2 + 100],
             [(img_size[1] / 10), img_size[0]],
             [(img_size[1] * 9 / 10), img_size[0]],
             [(img_size[1] / 2 + 100), img_size[0] / 2 + 100]])
        dst = np.float32(
            [[(img_size[1] / 4), 0],
             [(img_size[1] / 4), img_size[0]],
             [(img_size[1] * 3 / 4), img_size[0]],
             [(img_size[1] * 3 / 4), 0]])
        print(src,dst)
        topdown_image = self.perspective_transform(image,src,dst)
        if self.debug:
            visualizer.plot_perspective_transform(image,topdown_image,src,dst)

# Color & Gradient Threshold

# Perspective Transform