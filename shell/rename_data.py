#!/usr/bin/python3

import os, sys
from PIL import Image

path=sys.argv[1]

image_files=[".png", ".jpg", ".jpeg", ".gif", ".ps", ".eps", ".svg", ".pdf"]

myfiles = []
counter=1
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for f2 in f:
        image_check=[(t in f2) for t in image_files]
        fname_org=os.path.join(r, f2)
        #print(f2, image_check)
        if any(image_check):
            try:
                im=Image.open(fname_org)
                ftype=f2.split('.')[1]
                fname_new=r+"/"+str(counter)+"."+ftype
                print(fname_new)
                counter+=1
                os.rename(fname_org, fname_new)
            except OSError:
                os.remove(fname_org)
            except:
                pass
            
                




    
