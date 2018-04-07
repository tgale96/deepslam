import cv2
import os
from sys import argv
import matplotlib as mpl
import matplotlib.pyplot as plt

image_folder = argv[1]
out_folder = argv[2]
images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
images.sort()

def img_stats(img):
    print("max {} / min {} / mean {} / std {}"
          .format(img.max(), img.min(), img.mean(), img.std()))
    
for iname in images:
    img = cv2.imread(image_folder + "/" + iname, -1)

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, cmap='gray')
    fig.savefig(out_folder + iname)
    plt.close()
