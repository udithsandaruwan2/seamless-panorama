# Import Libraries
import os
import cv2
import math
import glob
import numpy as np
import matplotlib.pyplot as plt

# Helper functions for image enhancements

def equalize_histogram(image, alpha=0.5):
    """
    Apply histogram equalization to improve contrast of the image with reduced effect.
    
    Parameters:
        image (numpy.ndarray): Input image in RGB format.
        alpha (float): Weight for blending the equalized image with the original image.
                       0.0 = original image, 1.0 = fully equalized image.
    
    Returns:
        numpy.ndarray: Image with reduced contrast enhancement.
    """
    img_yuv = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    img_equalized = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
    
    # Blend equalized image with the original image
    img_output = cv2.addWeighted(image, 1 - alpha, img_equalized, alpha, 0)
    return img_output


def sharpen_image(image):
    """
    Sharpen the image to enhance edges and details.
    """
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    img_output = cv2.filter2D(image, -1, kernel)
    return img_output

def reduce_noise(image):
    """
    Reduce noise in the image using Non-Local Means Denoising.
    """
    img_output = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return img_output

def denoise(image):
    """
    Reduce noise in the image using Median Blur.
    :param image: Input image (BGR format)
    :return: Denoised image (BGR format)
    """
    median_denoise_img = cv2.medianBlur(image, 7)
    
    return median_denoise_img

def color_correction(image):
    """
    Correct colors in the image by equalizing the LAB color space luminance channel.
    """
    img_lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(img_lab)
    l = cv2.equalizeHist(l)
    img_lab = cv2.merge((l, a, b))
    img_output = cv2.cvtColor(img_lab, cv2.COLOR_LAB2RGB)
    return img_output

def is_blur(image, threshold=100.0):
    """
    Detect if the image is blurry using variance of Laplacian.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return laplacian_var < threshold

def fix_blur(image):
    """
    Fix blurry image by applying sharpening.
    """
    return sharpen_image(image)

# Apply enhancements to images

def enhance_images(images):
    """
    Apply a series of enhancements to a list of images.
    """
    enhanced_images = []
    for img in images:
        if is_blur(img):
            img = fix_blur(img)
        img = equalize_histogram(img)
        img = sharpen_image(img)
        img = reduce_noise(img)
        img = color_correction(img)
        enhanced_images.append(img)
    
    return enhanced_images

# Image stitching function

def Stitcher_create(images):
    """
    Stitch a list of images into a single panoramic image using SIFT.
    """
    # Initialize SIFT detector
    sift = cv2.SIFT_create()
    
    # List to store the keypoints and descriptors of all images
    keypoints = []
    descriptors = []

    # Detect keypoints and descriptors
    for img in images:
        kp, des = sift.detectAndCompute(img, None)
        keypoints.append(kp)
        descriptors.append(des)

    # Use BFMatcher to find the best matches
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(descriptors[0], descriptors[1])
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract location of good matches
    pts1 = np.float32([keypoints[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    pts2 = np.float32([keypoints[1][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Find homography
    h, mask = cv2.findHomography(pts2, pts1, cv2.RANSAC)

    # Warp the second image to align with the first
    height, width = images[0].shape[:2]
    result = cv2.warpPerspective(images[1], h, (width + images[1].shape[1], height))

    # Place the first image in the result
    result[0:height, 0:width] = images[0]

    # Crop the result to remove black edges
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(contours[0])
    result = result[y:y+h, x:x+w]

    return result

