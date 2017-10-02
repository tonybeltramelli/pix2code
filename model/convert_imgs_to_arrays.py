#!/usr/bin/env python
from __future__ import print_function
from __future__ import absolute_import
__author__ = 'Tony Beltramelli - www.tonybeltramelli.com'

import os
import sys
import shutil

from classes.Utils import *
from classes.model.Config import *

argv = sys.argv[1:]

if len(argv) < 2:
    print("Error: not enough argument supplied:")
    print("convert_imgs_to_arrays.py <input path> <output path>")
    exit(0)
else:
    input_path = argv[0]
    output_path = argv[1]

if not os.path.exists(output_path):
    os.makedirs(output_path)

print("Converting images to numpy arrays...")

for f in os.listdir(input_path):
    if f.find(".png") != -1:
        img = Utils.get_preprocessed_img("{}/{}".format(input_path, f), IMAGE_SIZE)
        file_name = f[:f.find(".png")]

        np.savez_compressed("{}/{}".format(output_path, file_name), features=img)
        retrieve = np.load("{}/{}.npz".format(output_path, file_name))["features"]

        assert np.array_equal(img, retrieve)

        shutil.copyfile("{}/{}.gui".format(input_path, file_name), "{}/{}.gui".format(output_path, file_name))

print("Numpy arrays saved in {}".format(output_path))
