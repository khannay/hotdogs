#!/usr/bash


echo "Randomly choose $3 of files from $1 and move to $2"

find $1 -type f -print0 | sort -Rz | cut -d $'\0' -f-$3 | xargs -0 mv -t $2

