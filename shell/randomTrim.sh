#!/usr/bash

echo "Randomly choose $2 of files to keep from a $1, delete the others"

find $1 -type f -print0 | sort -zR | tail -zn +$2 | xargs -0 rm
