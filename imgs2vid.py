import os
from sys import argv

os.system("ffmpeg -r 10 -i " + argv[1] + "/%d.png -vcodec mpeg4 -y movie.mp4")
