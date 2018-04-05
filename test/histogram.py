import cv2
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

def img_stats(t):
    t = np.array(t)
    print("max, min, mean, std: {} / {} / {} / {}"
          .format(t.max(), t.min(), t.mean(), t.std()))

iname = argv[1]
img = np.array(cv2.imread(iname, -1))
img = img.astype(np.float32) / 5000.
upper = int(img.max()+1)
print("upper: {}".format(upper))
hist = cv2.calcHist([img], [0], None, [100], [0, upper])

img_stats(img)

y = [i/100. * upper for i in range(100)]
plt.plot(y, hist)
plt.show()
