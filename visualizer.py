import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# # Create an image to draw the lines on
# warp_zero = np.zeros_like(warped).astype(np.uint8)
# color_warp = np.dstack((warp_zero, warp_zero, warp_zero))
#
# # Recast the x and y points into usable format for cv2.fillPoly()
# pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
# pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
# pts = np.hstack((pts_left, pts_right))
#
# # Draw the lane onto the warped blank image
# cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
#
# # Warp the blank back to original image space using inverse perspective matrix (Minv)
# newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))
# # Combine the result with the original image
# result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
# plt.imshow(result)

def plot_distortion_correction(distorted_image, undistorted_image, save_directory=''):
    # Plot original image and undistorted image
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(distorted_image)
    ax1.set_title('Distorted Image', fontsize=20)
    ax2.imshow(undistorted_image)
    ax2.set_title('Undistorted Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    if save_directory:
        plt.savefig(save_directory)


def plot_perspective_transform(image, topdownimage, src, dst):
    xs, ys = zip(*src)  # create lists of x and y values
    xd, yd = zip(*dst)  # create lists of x and y values
    print(xs,ys)
    print(xd,yd)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.add_patch(patches.Polygon(xy=list(src), color='r', linewidth=6, fill=False))
    ax1.set_title('Undistorted Image', fontsize=20)
    ax2.imshow(topdownimage)
    ax2.add_patch(patches.Polygon(xy=list(dst), color='g', linewidth=6, fill=False))
    ax2.set_title('Transformed Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()