"""
Creates hard clutter chips from ranges 4km-5km to train TCRNet-2 booster.
"""

import glob
from torch.utils.data import Dataset
from scipy.io import loadmat, savemat
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
import json
from skimage.transform import resize
# import tcr_cnn
import tcr_cnn_2streams as tcr_cnn
import matplotlib.patches as patches
from PIL import Image
import matplotlib
import pandas as pd
import copy
import imresize

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def get_detections(input_image,ndetects):
    image = input_image.copy()
    minval = image.min()
    nrows,ncols = image.shape
    confs=[]
    row_dets=[]
    col_dets=[]
    for i in range(ndetects):
        row,col = np.unravel_index(image.argmax(), image.shape)
        val = image[row,col]
        r1 = max(row - 10, 0)
        r2 = min(r1 + 19, nrows)
        r1 = r2 - 19
        c1 = max(col - 20, 1)
        c2 = min(c1 + 39, ncols)
        c1 = c2 - 39
        image[r1: r2+1, c1:c2+1]=np.ones((20, 40)) * minval;
        confs.append(val)
        row_dets.append(row)
        col_dets.append(col)

    confs = np.array(confs)
    Aah = confs.std() * 6** .5 / 3.14158
    cent = confs.mean() - Aah * 0.577216649;
    confs = (confs - cent) / Aah;


    row_dets = np.array(row_dets)
    col_dets = np.array(col_dets)

    return confs, row_dets, col_dets

def show(im):
    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(im)
    plt.show()

def Within20x40Region(blx, bly, urx, ury, c2, r2):
    '''
    blx, bly, urx, ury are the bottom left and upper right coordinates of the detection with the largest score.
    c2, r2 are the coordinates of the detection that is being check if it's within a 20 x 40 region centered on the first detection.
    '''

    if (c2 > blx and c2 < urx and r2 > bly and r2 < ury):
        return True
    else:
        return False

def convert(input_img, out_img_min, out_img_max, out_data_type):
    input_min = input_img.min()
    input_max = input_img.max()

    a = (out_img_max - out_img_min)/(input_max - input_min)
    b = out_img_max - a*input_max
    out_img = (a*input_img + b).astype(out_data_type)
    return out_img


# train json files
samples  = json.load(open('../data/train_4_5.json'))

# input location of full-size images
imgdir = '/data/NVESD/matlab_1_5/'

net = tcr_cnn.tcrNet().to(device)
net.load_state_dict(torch.load('./weights_filters/tcrnet2_weights.pth'))

index=0
dets=[]
fas=[]
nframes=0
ntgt=0

# Chip_info = pd.DataFrame(columns=["FileName", "Name", "Frame", "Chip", "Target_Category", "Score", "Det_x_coord", "Det_y_coord", "GT_x_coord", "GT_y_coord", "mat_path"])
# df_json = []

#Output data location
outputdir = '../data/train/chips40x80/hard_clutter/'

for sample in samples:
    print(sample['name'] + "_" + sample['frame'])
    imfile = imgdir + sample['name'] + '_' + sample['frame'] + '.mat'
    im = loadmat(imfile)['image']
    target_range = sample['range'] * 1000
    scale_factor = target_range/2500
    # im = scale(im,scale_factor)
    im = imresize.imresize(im,scale_factor, method='bilinear')
    nrows, ncols = im.shape
    im = torch.tensor(im).unsqueeze(0).unsqueeze(0).float().to(device)
    output = net(im)
    output = output.cpu().detach()[0,0,:,:].numpy()

    output = pad(output,nrows,ncols)
    confs, row_dets, col_dets = get_detections(output,5)
    row_dets = row_dets/scale_factor
    col_dets = col_dets/scale_factor

    targets = sample['targets']
    nt = len(targets)
    ndets = confs.shape[0]
    ntgt += nt
    nframes += 1

   

    image = loadmat(imfile)['image']
    # image = scale(image,scale_factor)
    image = imresize.imresize(image,scale_factor, method='bilinear')

    w = 40
    h = 20

    ymax = image.shape[0]
    xmax = image.shape[1]

    for target in targets:
        r = target['center'][1]
        c = target['center'][0]
        
        foundtgt = np.zeros(ndets)
        tmpdets = []
        tmpfas = []
        tmpcol_dets = []
        tmprow_dets = []

        for i in range(ndets):
            dist = ((r - row_dets[i]) ** 2 + (c - col_dets[i]) ** 2)**.5

            if dist < 20:
                foundtgt[i] = 1
                tmpdets.append(confs[i])

        if len(tmpdets) >= 1:
            dets.append(max(tmpdets))
            T = np.where(confs == max(tmpdets))[0]
            Tcol_det = col_dets[T]
            Trow_det = row_dets[T]

            # # Create the target chip (1 chip)
            # targetfilepath_mat = outputdir + sample['name'] + '_' + sample['frame'] + '_target_' + target['category'] + '.mat'
            # # targetfilepath_png = outputdir + 'png_scaled/targets/' + sample['name'] + '_' + sample['frame'] + '_target_' + target['category'] + '.png'
            # cx = Tcol_det * scale_factor
            # cy = Trow_det * scale_factor


            # ylow = int(cy - h)
            # if ylow < 0:
            #     ylow = 0

            # yhi = int(cy + h)
            # if yhi >= ymax:
            #     yhi = ymax
            # xlow = int(cx - w)

            # if xlow < 0:
            #     xlow = 0

            # xhi = int(cx + w)
            # if xhi >= xmax:
            #     xhi = xmax

            # target_chip = image[ylow: yhi, xlow: xhi]

            # if (target_chip.shape[0] > 2*h or target_chip.shape[1] > 2*w):
            #     target_chip = target_chip[0:2*h,0:2*w]
            # elif (target_chip.shape[0] == 2*h and target_chip.shape[1] == 2*w):
            #     target_chip = target_chip
            # else:
            #     target_chip = pad(target_chip,2*h,2*w)

            # trgt_filename = targetfilepath_mat.split("/")[-1].split(".")[0]

            # # Create mat chips
            # savemat(targetfilepath_mat, {"chip":target_chip})

            # ## Create Pandas DataFrame
            # Chip_info = Chip_info.append({"FileName": trgt_filename, "Name": sample['name'], "Frame": sample['frame'], "Chip": "target", "Target_Category": target['category'], "Score": max(tmpdets), "Det_x_coord": Tcol_det[0], "Det_y_coord": Trow_det[0], "GT_x_coord": c, "GT_y_coord": r, "mat_path": outputdir + trgt_filename + ".mat"} , ignore_index=True)


            if len(tmpdets) >= 2:
                tmpdets2 = [n for n in tmpdets if n != max(tmpdets)]
                T2 = np.where(np.in1d(confs, tmpdets2))[0]

                T2col_det = col_dets[T2]
                T2row_det = row_dets[T2]

                # 20x40 rectangle
                blx = Tcol_det - 20
                bly = Trow_det - 10
                urx = Tcol_det + 20
                ury = Trow_det + 10

                for m in range(len(tmpdets2)):
                    if Within20x40Region(blx, bly, urx, ury, T2col_det[m], T2row_det[m]):
                        T3 = np.where(confs == tmpdets2[m])[0]
                        foundtgt[T3] = 2

                    else:
                        T3 = np.where(confs == tmpdets2[m])[0]
                        foundtgt[T3] = 0

        I = np.where(foundtgt == 0)[0]
        for indx_a in confs[I]:
            fas.append(indx_a)
            tmpfas.append(indx_a)
        for indx_b in col_dets[I]:
            tmpcol_dets.append(indx_b)
        for indx_d in row_dets[I]:
            tmprow_dets.append(indx_d)


        for k in range(len(tmpfas)):
            clutterfilename_mat = outputdir + sample['name'] + '_' + sample['frame'] + '_clutter_' + str(k+1) + "_" + target['category'] + '.mat'

            cx = tmpcol_dets[k] * scale_factor
            cy = tmprow_dets[k] * scale_factor


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

            clutter_chip = image[ylow: yhi, xlow: xhi]

            if (clutter_chip.shape[0] > 2*h or clutter_chip.shape[1] > 2*w):
                clutter_chip = clutter_chip[0:2*h,0:2*w]
            elif (clutter_chip.shape[0] == 2*h and clutter_chip.shape[1] == 2*w):
                clutter_chip = clutter_chip
            else:
                clutter_chip = pad(clutter_chip,2*h,2*w)


            cltr_filename = clutterfilename_mat.split("/")[-1].split(".")[0]

            # save clutter chips in mat and png formats
            # Create mat chips
            savemat(clutterfilename_mat, {"clutter_chip":clutter_chip})

            # # Create png images
            # # scale the chip from 1-255 and convert to int8
            # clutter_chip = convert(clutter_chip, 1, 255, np.uint8)
            # clutter_chip = Image.fromarray(clutter_chip)
            # clutter_chip.save(clutterfilename_png)
#             matplotlib.image.imsave(clutterfilename_png, clutter_chip)

            ## Create Pandas DataFrame with chip info
            # Chip_info = Chip_info.append({"FileName": cltr_filename , "Name": sample['name'], "Frame": sample['frame'], "Chip": "CLUTTER", "Target_Category": "CLUTTER", "Score": tmpfas[k], "Det_x_coord": tmpcol_dets[k], "Det_y_coord": tmprow_dets[k], "GT_x_coord": np.nan, "GT_y_coord": np.nan, "mat_path": outputdir + cltr_filename + ".mat"} , ignore_index=True)

            # ## Create JSON File
            # tmp_clutter_json = copy.deepcopy(sample)
            # tmp_clutter_json['targets'] = [copy.deepcopy(target)]
            # tmp_clutter_json['targets'][0]['filename'] = cltr_filename
            # tmp_clutter_json['targets'][0]['score'] = tmpfas[k]
            # tmp_clutter_json['targets'][0]['det_center'] = [tmpcol_dets[k], tmprow_dets[k]]
            # tmp_clutter_json['targets'][0]['chip'] = 'clutter'

            # df_json.append(copy.deepcopy(tmp_clutter_json))




# Ex port CSV and JSON file
# Chip_info.to_csv("../data/test_chip_info.csv", index=False)

# with open('../data/train_180samples_scaled2500.json', 'w') as outfile:
#     json.dump(df_json, outfile)

print("Chips are created successfully.")

# dets = np.array(dets)
# np.save('../data/dets',dets)
# fas = np.array(fas)
# np.save('../data/fas',fas)
# ntgt = np.array([ntgt])
# np.save('../data/ntgt',ntgt)
# nframes = np.array([nframes])
# np.save('../data/nframes',nframes)
# print(ntgt,nframes,len(dets),len(fas))
# FAs dets Ntgt Nframes
