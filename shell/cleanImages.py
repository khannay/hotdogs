#!/usr/bin/python3

import os
from PIL import Image
import sys

direct=sys.argv[1]

myfiles=os.listdir(direct)

errors=[]
for f in myfiles:
    try:
        im=Image.open(direct+f)
    except OSError:
        errors.append(f)
    except:
        pass

print(errors)

for e in errors:
    try:
        os.remove(direct+e)
    except:
        print("Error with ", e)
