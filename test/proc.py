import cv2
from sys import argv
import numpy as np

# rgb image file name
iname = argv[1]
img = cv2.imread(iname, -1)

dname = argv[2]
depth = cv2.imread(dname, -1)

# kernel and threshold parameters 
k = 5
t = 50

def unsharp_mask(img):
    blur = cv2.GaussianBlur(img, (15, 15), sigmaX = 1500.0, sigmaY = 1500.0)
    alpha = .75
    sharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)    
    k = 11
    kernel = np.ones((k, k), np.float32) / (k*k)
    filtered = cv2.filter2D(sharp, -1, kernel)

def edge_det(img, t):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (k, k), 0)

    # Don't threshold
    # (t, binary) = cv2.threshold(blur, t, 255, cv2.THRESH_BINARY_INV)
    binary = blur
    
    # perform Sobel edge detection in x and y dimensions
    edgeX = cv2.Sobel(binary, cv2.CV_16S, 1, 0)
    edgeY = cv2.Sobel(binary, cv2.CV_16S, 0, 1)

    # convert back to 8-bit, unsigned numbers and combine edge images
    edgeX = np.uint8(np.absolute(edgeX))
    edgeY = np.uint8(np.absolute(edgeY))
    edge = cv2.bitwise_or(edgeX, edgeY)

    # DEBUG: thresh after edge
    (t, binary) = cv2.threshold(edge, t, 255, cv2.THRESH_BINARY)
    return binary
    # return edge

def contours(img):
    tmp = img.copy()
    return cv2.findContours(tmp, cv2.RETR_LIST,
                            cv2.CHAIN_APPROX_SIMPLE)
    
# find the edges & contours
# edges = edge_det(img, t)
# contour, hierarchy = contours(edges)
        
# for i in range(len(contour)):
#     print(contour[i].shape)
#     if contour[i].shape[0] > t:
#         blank = np.zeros(edges.shape, dtype=np.uint8)
#         cv2.drawContours(blank, contour, i, 255, -1)
#         cv2.imshow('contour', blank)
#         cv2.waitKey()

def get_masks(img):
    edges = edge_det(img, t)
    contour, hierarchy = contours(edges)
    masks = []
    for i in range(len(contour)):
        if contour[i].shape[0] > t:
            blank = np.zeros(edges.shape, dtype=np.uint8)
            cv2.drawContours(blank, contour, i, 1, -1)
            masks.append(blank)
    return masks

def show_bin(s, img):
    test = img.copy()
    test[test == 1] = 255
    cv2.imshow(s, test)

cv2.imshow("depth", depth)
masks = get_masks(img)
for mask in masks:
    (_, inv_mask) = cv2.threshold(mask, 0, 1,
                                  cv2.THRESH_BINARY_INV)
    masked = np.multiply(mask, depth)
    rest = np.multiply(inv_mask, depth)


    # cv2.imshow("depth", depth)
    # show_bin("mask", mask)
    # show_bin("inv_mask", inv_mask)
    # cv2.imshow("masked", masked)
    # cv2.imshow("inv_masked", rest)

    # filter the selected region and add back
    kernel = np.ones((k, k), np.float32) / (k*k)
    filtered = cv2.filter2D(masked, -1, kernel)
    depth = filtered + rest
    # cv2.imshow("new_depth", depth)
    # cv2.waitKey()

cv2.imshow("new_depth", depth)
cv2.waitKey()
