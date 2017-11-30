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

def plot_two_images(image1, image2, title, save_directory=''):
    # Plot original image and undistorted image
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1, cmap='gray')
    ax1.set_title('Image before ' + title, fontsize=20)
    ax2.imshow(image2, cmap='gray')
    ax2.set_title('Image after ' + title, fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    if save_directory:
        plt.savefig(save_directory)

def plot_polynomial_fit(image1, image2, left_fitx, right_fitx, ploty, save_directory=''):
    # Plot original image and undistorted image
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image1, cmap='gray')
    ax1.set_title('Warped image', fontsize=20)
    ax2.imshow(image2, cmap='gray')
    ax2.plot(left_fitx, ploty, color='yellow')
    ax2.plot(right_fitx, ploty, color='yellow')
    # ax2.xlim(0, 1280)
    # ax2.ylim(720, 0)
    ax2.set_title('Found polynomial', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    if save_directory:
        plt.savefig(save_directory)

def plot_two_times_four_images(images, save_directory=''):
    # Plot original image and undistorted image
    f, axarr = plt.subplots(2, 4, figsize=(24, 9))
    f.tight_layout()
    axarr[0, 0].imshow(images[0])
    axarr[0, 0].set_title('Original', fontsize=20)
    axarr[0, 1].imshow(images[1], cmap='gray')
    axarr[0, 1].set_title('1', fontsize=20)
    axarr[0, 2].imshow(images[2], cmap='gray')
    axarr[0, 2].set_title('2', fontsize=20)
    axarr[0, 3].imshow(images[3], cmap='gray')
    axarr[0, 3].set_title('3', fontsize=20)
    axarr[1, 0].imshow(images[4], cmap='gray')
    axarr[1, 0].set_title('Result', fontsize=20)
    axarr[1, 1].imshow(images[5], cmap='gray')
    axarr[1, 1].set_title('4', fontsize=20)
    axarr[1, 2].imshow(images[6], cmap='gray')
    axarr[1, 2].set_title('5', fontsize=20)
    axarr[1, 3].imshow(images[7], cmap='gray')
    axarr[1, 3].set_title('6', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()
    if save_directory:
        plt.savefig(save_directory)


def plot_perspective_transform(image, topdownimage, src, dst):
    xs, ys = zip(*src)  # create lists of x and y values
    xd, yd = zip(*dst)  # create lists of x and y values
    # print(xs,ys)
    # print(xd,yd)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image, cmap='gray')
    ax1.add_patch(patches.Polygon(xy=list(src), color='r', linewidth=2, fill=False))
    ax1.set_title('Undistorted Image', fontsize=20)
    ax2.imshow(topdownimage, cmap='gray')
    ax2.add_patch(patches.Polygon(xy=list(dst), color='g', linewidth=2, fill=False))
    ax2.set_title('Transformed Image', fontsize=20)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def plot_histogram(histogram):
    plt.plot(histogram)
    plt.show()