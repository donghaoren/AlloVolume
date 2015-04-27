#!/usr/bin/env python

from allovolume import VolumeFrom3DArray
from scipy import misc
import numpy as npi
import sys

if len(sys.argv) < 3:
    print "Usage: images2volume.py output images..."
    print "Note: I'll sort the images by their names."
    exit(0)

output_file = sys.argv[1]
images = sorted(sys.argv[2:])

print "Writing output to `%s`, with %d slices." % (output_file, len(images))

d = misc.imread(images[0])
xsize = d.shape[0]
ysize = d.shape[1]
zsize = len(images)

print "Volume size: %dx%dx%d." % (xsize, ysize, zsize)

volume = np.zeros((xsize, ysize, zsize))

for i in range(len(images)):
    d = misc.imread(images[i])
    volume[:,:,i] = d

with open(output_file, "wb") as fout:
    VolumeFrom3DArray(volume, (-xsize / 2.0, -ysize / 2.0, -zsize / 2.0), (xsize / 2.0, ysize / 2.0, zsize / 2.0), fout)
