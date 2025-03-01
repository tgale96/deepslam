import cv2
from sys import argv
import numpy as np

# rgb image file name
iname = argv[1]
img = cv2.imread(iname, -1)

dname = argv[2]
depth = cv2.imread(dname, -1)

# kernel and threshold parameters 
edge_k = 5
mean_k = 45
t = 50

def edge_det(img, t):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(img, (edge_k, edge_k), 0)

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
    
def show_bin(s, img):
    test = img.copy()
    test[test == 1] = 255
    cv2.imshow(s, test)

def get_masks(img):
    edges = edge_det(img, t)
    _, contour, hierarchy = contours(edges)
    masks = []
    whole_mask = np.zeros(edges.shape, dtype=np.uint8)
    for i in range(len(contour)):
        if contour[i].shape[0] > t:
            blank = np.zeros(edges.shape, dtype=np.uint8)
            cv2.drawContours(blank, contour, i, 1, -1)
            whole_mask = cv2.bitwise_or(whole_mask, blank)
            masks.append(blank)
            
    # show_bin("whole_mask", whole_mask)
    return masks, whole_mask

# mask out anything lower than mean - std
depth = np.array(depth)
test = depth[(depth < (depth.mean() - depth.std()))]
print("thresholding {} points out of {}".format(test.size, depth.size))
depth[depth < (depth.mean() - depth.std())] = 0.0

# Get the segmentation masks
masks, whole_mask = get_masks(img)
(_, inv_whole_mask) = cv2.threshold(whole_mask, 0, 1, 
                                    cv2.THRESH_BINARY_INV)
masks.append(inv_whole_mask)

# Sort the masks from smallest to largest
masks.sort(key = lambda x: x[x==1].size)

# Mask for keeping track of filtered regions
used = np.zeros(depth.shape, dtype=np.uint8)
for mask in masks:
    # subtract the used mask 
    mask = cv2.bitwise_and(mask, cv2.bitwise_not(used))
    (_, inv_mask) = cv2.threshold(mask, 0, 1,
                                  cv2.THRESH_BINARY_INV)
    masked = np.multiply(mask, depth)
    rest = np.multiply(inv_mask, depth)
    
    # Keep track of the pixels we have already filtered
    used = cv2.bitwise_or(used, mask)

    # DEBUG: set region to mean
    # mean = masked[masked != 0.0].mean() 
    # masked[masked != 0.0] = mean
    # filtered = masked.copy()
    
    # set background of the image to the mean of the region
    mean = masked[masked != 0.0].mean()
    masked[masked == 0.0] = mean
    
    # filter the selected region and add back
    kernel = np.ones((mean_k, mean_k), np.float32) / (mean_k**2)
    filtered = cv2.filter2D(masked, -1, kernel)
    
    # DEBUG: try median filtering for lower variance
    filtered = cv2.medianBlur(filtered, 5)

    # mask out the background again
    filtered = np.multiply(mask, filtered)

    # show_bin("mask", mask)
    # show_bin("inv_mask", inv_mask)
    # show_bin("used", used)
    # cv2.imshow("filt", filtered)
    # cv2.waitKey()

    # add the rest of the image back
    depth = filtered + rest

# DEBUG: mask out parts of image not filtered
dbg = np.multiply(whole_mask, depth)

show_bin("whole_mask", whole_mask)
cv2.imshow("debug", dbg)
cv2.imshow("new_depth", depth)
cv2.waitKey()

# DEBUG
cv2.imwrite("new_depth.png", depth)
# cv2.imwrite("new_depth.png", dbg)
