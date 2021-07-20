"""
Crops 40x80 chips to 20x40 chips to generate qcf filters.
"""

## Crop Images
import glob
from scipy.io import loadmat, savemat

def crop(input_img,d1,d2):
    '''
    This function returns a cropped image.

    input_img  = input image
    d1 = rows of cropped image
    d2 = column of cropped image
    '''
    m,n = input_img.shape

    off1 = round((m-d1)/2)
    off2 = round((n-d2)/2)

    if off1 < 1:
        off1 = 1

    if off2 < 1:
        off2 = 1

    return input_img[off1:off1+d1,off2:off2+d2]


def make_20x40_chip(files, chip, outpath):
    for i in range(len(files)):
        image_path = files[i]
        image = loadmat(image_path)[chip]
        image_cropped = crop(image, 20, 40)
        
        filename = outpath + files[i].split("/")[-1]
        savemat(filename, {chip:image_cropped})
    print(chip + 's cropped to 20x40')

# target chips crop from 40x80 to 20x40
inputpath = '../data/train/chips40x80/targets/'
outpath = '../data/train/chips20x40/targets/'
target_files  = glob.glob(inputpath + '*.mat')
make_20x40_chip(files=target_files, chip='target_chip', outpath=outpath)

# clutter chips crop from 40x80 to 20x40
inputpath = '../data/train/chips40x80/clutter/'
outpath = '../data/train/chips20x40/clutter/'
clutter_files  = glob.glob(inputpath + '*.mat')
make_20x40_chip(files=clutter_files, chip='clutter_chip', outpath=outpath)