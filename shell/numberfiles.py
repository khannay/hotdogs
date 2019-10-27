#!/usr/bin/python3

"""Just recurse and name all files in a directory as num.jpg"""

import os, sys
from PIL import Image

path=sys.argv[1]

image_files=[".png", ".jpg", ".jpeg", ".gif", ".ps", ".eps", ".svg", ".pdf", ".JPG", ".JPEG", ".PNG"]

myfiles = []
counter=1
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for f2 in f:
        fname_org=os.path.join(r, f2)
        ftype="jpg" 
        fname_new=r+"/"+str(counter)+"."+ftype
        print(fname_new)
        counter+=1
        os.rename(fname_org, fname_new)
            
