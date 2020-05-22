#############################################################################
# Image matching for 3D printing to detect failures
#
# Description: Images are loaded from disk, applied thresholding and
#   computed hu moments to calculate the difference between them
#
# References:
#   - Computer Vision Course 2020, Laboratory 5, from Technical University
#       Iasi, Faculty of Automatic Control and Computer Engineering
#############################################################################

import cv2
import os
import glob

from scipy.spatial import distance


IMG_SLICER = 0
IMG_PRINTED = 1


def load_images(path):
    data_path = os.path.join(path,'*g')
    files = glob.glob(data_path)
    data = []
    for f in files:
        images = dict()
        images["name"] = os.path.basename(f)
        img = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        images["image"] = img
        data.append(images)

    return data


def load_image(path):
    image = dict()
    image["name"] = os.path.basename(path)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    image["image"] = img

    return image


def draw_image(image):
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Image', 600, 600)
    cv2.imshow('Image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def calculate_hu_moments(image, threshold, type):
    filtered_image = cv2.medianBlur(image, 5)
    _, im = cv2.threshold(filtered_image, threshold, 255, cv2.THRESH_BINARY_INV if type==IMG_SLICER else cv2.THRESH_BINARY)
    draw_image(im)
    moments = cv2.moments(im)
    hu_moments = cv2.HuMoments(moments)

    return hu_moments


def compare_moments(moment_1, moment_2):
    difference = distance.euclidean(moment_1, moment_2)

    return difference


def successful_test():
    # Read test and database images
    slicer_images = load_images(".images/phil_slicer/")
    printer_images = load_images(".images/phil_3d_printed/")

    for sl_img, pr_img in zip(slicer_images, printer_images):
        sl_hu_moments = calculate_hu_moments(sl_img["image"], 127, IMG_SLICER)
        pr_hu_moments = calculate_hu_moments(pr_img["image"], 170, IMG_PRINTED)
        diff = compare_moments(sl_hu_moments, pr_hu_moments)
        print("Similarity(" + sl_img["name"] + ", " + pr_img["name"] + " = " + str(diff))


def fail_test():
    # Read test and database images
    slicer_image = load_image(".images/phil_slicer/phil_layer2_slicer.png")
    successful_image = load_image(".images/phil_3d_printed/phil_layer2_printed.jpg")
    failed_images = load_images(".images/phil_3d_printed/fault/")

    sl_hu_moments = calculate_hu_moments(slicer_image["image"], 127, IMG_SLICER)
    su_hu_moments = calculate_hu_moments(successful_image["image"], 230, IMG_PRINTED)
    diff = compare_moments(sl_hu_moments, su_hu_moments)
    print("Similarity(" + slicer_image["name"] + ", " + successful_image["name"] + " = " + str(diff))
    for fl_img in failed_images:
        pr_hu_moments = calculate_hu_moments(fl_img["image"], 230, IMG_PRINTED)
        diff = compare_moments(sl_hu_moments, pr_hu_moments)
        print("Similarity(" + slicer_image["name"] + ", " + fl_img["name"] + " = " + str(diff))


def nothing(x):
    pass


def slider_calibration():
    image = load_image(".images/phil_3d_printed/phil_layer2_printed.jpg")

    cv2.namedWindow("image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 600, 600)
    cv2.createTrackbar('t', 'image', 0, 255, nothing)

    img = cv2.medianBlur(image["image"], 7)
    ret, th = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    while True:
        cv2.imshow('image', th)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

        threshold = cv2.getTrackbarPos('t', 'image')
        ret, th = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)

    cv2.destroyAllWindows()


def main():
    #successful_test()
    fail_test()

if __name__== "__main__":
    main()