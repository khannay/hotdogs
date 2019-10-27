#!/usr/bin/python3

"""
Will walk through a directory and all of its subdierctories and rename all image files it finds to number.format
If any image files cannot be opened by PIL then it will remove those images from the directories as well.

Will also remove any duplicated files from the directory and its subs
"""

import os, sys
from PIL import Image
import uuid


path=sys.argv[1]

print("Removing any duplicate files from the directory and its subs using fdupes")
remove_dup="fdupes -rdN "+path
os.system(remove_dup)



image_files=[".png", ".jpg", ".jpeg", ".gif", ".ps", ".eps", ".svg", ".pdf", ".JPG", ".JPEG", ".PNG"]

myfiles = []
counter=1
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
    for f2 in f:
        counter+=1
        image_check=[(t in f2) for t in image_files]
        fname_org=os.path.join(r, f2)
        if any(image_check):
            try:
                im=Image.open(fname_org)
                ftype=f2.split('.')[-1]
                unique_filename=str(uuid.uuid4())
                fname_new=r+"/"+unique_filename+"."+ftype
                #print(fname_new)
                os.rename(fname_org, fname_new)
            except OSError:
                print("Removing file as it caused an exception: ", fname_org)
                os.remove(fname_org)
            except:
                print("Error with file: ", fname_org, " nothing done about this")
                pass
            






    
