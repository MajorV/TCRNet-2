'''
This script creates target ground truth chips and random clutter chips of 40x80 size for all frames and scenarios.
* Change w and h in the script to generate 20x40 chips or crop 40x80 chips at center.
* create chips40x80/targets/ and chips40x80/clutter/ in data folder.
'''

import pickle
from scipy.io import loadmat, savemat
import numpy as np
import json
import random
import imresize
from skimage.transform import resize

def scale(image, factor):
    x, y = image.shape
    x = int(round(factor * x))
    y = int(round(factor * y))
    return resize(image,(x, y))

def pad(image,nrows,ncols):
    nrows = nrows
    ncols = ncols
    y, x = image.shape
    y_pad = (nrows-y)
    x_pad = (ncols-x)
    return np.pad(image,((y_pad//2, y_pad//2 + y_pad%2), 
                     (x_pad//2, x_pad//2 + x_pad%2)),
                  mode = 'constant')

def make_chips(samples, inputdir, outputdir):
        instances = []
        print('number of frames', len(samples))
        targets = 0
        targets2 = 0
        for sample in samples:
            print(sample['name'] + "_" + sample['frame'])
            for target in sample['targets']:
                targets += 1
                if target['inst_id'] not in instances:
                    targets2 += 1
                    instances.append(target['inst_id'])
                    infilename = inputdir + sample['name'] + '_' + sample['frame'] + '.mat'
                    
                    # scale images to 2500m
                    target_range = sample['range'] * 1000
                    scale_factor = target_range/2500
                    image = loadmat(infilename)['image']
                    image = imresize.imresize(image,scale_factor, method='bilinear')
                    # image = scale(image,scale_factor)
                    cx = target['center'][0] * scale_factor
                    cy = target['center'][1] * scale_factor
                    targetfilename = outputdir + 'chips40x80/targets/' + sample['name'] + '_' + sample['frame'] + '_target_' + target['category'] + '.mat'
                    

                    ulx = target['ul'][0]
                    uly = target['ul'][1]

                    if ulx < 0:
                        ulx = 0
                        print('problem', sample['info'])
                    if uly < 0:
                        uly = 0
                        print('problem', sample['info'])

                    w = 40
                    h = 20

                    ymax = image.shape[0]
                    xmax = image.shape[1]

                    ylow = int(cy - h)
                    if ylow < 0:
                        ylow = 0

                    yhi = int(cy + h)
                    if yhi >= ymax:
                        yhi = ymax
                    xlow = int(cx - w)

                    if xlow < 0:
                        xlow = 0

                    xhi = int(cx + w)
                    if xhi >= xmax:
                        xhi = xmax

                    target_chip = image[ylow: yhi, xlow: xhi]

                    # make sure to keep the chip size 40x80/2hx2w
                    if (target_chip.shape[0] > 2*h or target_chip.shape[1] > 2*w):
                        target_chip = target_chip[0:2*h,0:2*w]
                    elif (target_chip.shape[0] == 2*h and target_chip.shape[1] == 2*w):
                        target_chip = target_chip
                    else:
                        target_chip = pad(target_chip,2*h,2*w)


                    savemat(targetfilename, {"target_chip":target_chip})
                    

                    ## to create random clutter
                    ncx = random.randint(w, xmax - 1 - w)
                    ncy = random.randint(h, ymax - 1 - h)

                    while ((ncx - cx) ** 2 + (ncy - cy) ** 2) ** .5 < 90:
                        ncx = random.randint(w, xmax - 1 - w)
                        ncy = random.randint(h, ymax - 1 - h)
                    nylow = ncy - h
                    nyhi = ncy + h
                    nxlow = ncx - w
                    nxhi = ncx + w
                    clutter_chip = image[nylow: nyhi, nxlow: nxhi]

                    # make sure to keep the chip size 40x80/2hx2w
                    if (clutter_chip.shape[0] > 2*h or clutter_chip.shape[1] > 2*w):
                        clutter_chip = clutter_chip[0:2*h,0:2*w]
                    elif (clutter_chip.shape[0] == 2*h and clutter_chip.shape[1] == 2*w):
                        clutter_chip = clutter_chip
                    else:
                        clutter_chip = pad(clutter_chip,2*h,2*w)

                    clutterfilename = outputdir + 'chips40x80/clutter/' + sample['name'] + '_clutter_' + sample['frame'] + '_' + \
                                      target['category'] + '.mat'
                    savemat(clutterfilename, {"clutter_chip":clutter_chip})
        # print('targets1', targets)
        # print('targets2', targets2)


# location of full size frames
inputdir = "/data/NVESD/matlab_1_5/"
# inputdir = "../data/matlab_fullsize_example/"

# output location
outputdir = "../data/train/"

# file containing information of all frames i.e. target location
master_file  = json.load(open('../data/train_1to2.json'))
# master_file  = json.load(open('../data/train_1to2_example.json'))


# create chips 40x80 (scaled)
make_chips(samples = master_file, inputdir = inputdir, outputdir = outputdir)