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
from scipy.io import loadmat
from skimage.transform import resize
import matplotlib.patches as patches
from PIL import Image
import matplotlib
import pandas as pd
import copy
import imresize
# import tcr_cnn
import tcr_cnn_2streams as tcr_cnn
import crop


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def pad(image,nrows,ncols):
    out = np.zeros((nrows,ncols))
    m, n = image.shape
    o1 = nrows/2 + 1
    o2 = ncols/2 + 1
    r1 = int(round(o1 - m/2))
    r2 = int(round(r1 + m - 1))
    c1 = int(round(o2 - n/2))
    c2 = int(round(c1 + n -1))
    out[r1:r2+1,c1:c2+1] = image
    return out

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
        image[r1: r2+1, c1:c2+1]=np.ones((20, 40)) * minval
        confs.append(val)
        row_dets.append(row)
        col_dets.append(col)

    confs = np.array(confs)
    Aah = confs.std() * np.sqrt(6) / np.pi
    cent = confs.mean() - Aah * 0.577216649
    confs = (confs - cent) / Aah

    row_dets = np.array(row_dets)
    col_dets = np.array(col_dets)

    return confs, row_dets, col_dets


def Within20x40Region(blx, bly, urx, ury, c2, r2):
    '''
    blx, bly, urx, ury are the bottom left and upper right coordinates of the detection with the largest score.
    c2, r2 are the coordinates of the detection that is being check if it's within a 20 x 40 region centered on the first detection.
    '''

    if (c2 > blx and c2 < urx and r2 > bly and r2 < ury):
        return True
    else:
        return False
        
# load json file with test images information
samples  = json.load(open('../data/test_25to35all_example.json'))

# location of test images
imgdir = '../data/test/'


# Load the primary TCRNet
net = tcr_cnn.tcrNet().to(device)
net.load_state_dict(torch.load('./weights_filters/tcrnet2_weights.pth'))
net.eval()

# Load the second TCRNet for boosting
net2 = tcr_cnn.tcrNet().to(device)
net2.load_state_dict(torch.load('./weights_filters/tcrnet2_booster_weights.pth'))
net2.eval()


dets=[]
fas=[]
nframes=0
ntgt=0

dets2=[]
fas2=[]

for sample in samples:
    # print(sample['name'] + "_" + sample['frame'])
    imfile = imgdir + sample['name'] + '_' + sample['frame'] + '.mat'
    im = loadmat(imfile)['image']

    target_range = sample['range'] * 1000
    scale_factor = target_range/2500

    im = imresize.imresize(im,scale_factor, method='bilinear')
    nrows, ncols = im.shape

    # if nrows<=512:
    #     img=np.zeros((512,640))
    #     img[0:nrows,0:ncols] = im-im.mean()
    # else:
    #     img=crop.crop(im,512,640)
    #     img=img-img.mean()

    im=im-im.mean()

    im = torch.tensor(im).unsqueeze(0).unsqueeze(0).float().to(device)

    # TCRNet-2 
    output = net(im)
    output = output**2
    output = output.cpu().detach()[0,0,:,:].numpy()
    Y = pad(output,nrows,ncols)

    # TCRNet 2
    output2 = net2(im)
    output2 = output2**2
    output2 = output2.cpu().detach()[0,0,:,:].numpy()
    Y2 = pad(output2,nrows,ncols)

    # Detections
    ndets = 10
    confs, row_dets, col_dets = get_detections(Y,ndets)

    N=len(confs)

    # if N>5:
    #     tconfs=confs[6:N]
    # else:
    #     tconfs=confs

    L = []
    for i in range(ndets):

        r=row_dets[i]
        c=col_dets[i]

        r1 = max(r - 3, 0)
        r2 = min(r1 + 6, 512)
        r1 = r2 - 6
        c1 = max(c - 3, 0)
        c2 = min(c1 + 6, 640)
        c1 = c2 - 6

        r1 = np.rint(r1).astype(np.int32)
        r2 = np.rint(r2).astype(np.int32)
        c1 = np.rint(c1).astype(np.int32)
        c2 = np.rint(c2).astype(np.int32)

        v = np.amax(Y2[r1:r2+1, c1:c2+1], axis = 0)
        L.append(v)

    L = np.asarray(L)

    Aah = np.std(L, axis = 0) * np.sqrt(6)/np.pi
    cent = np.mean(L, axis = 0) - Aah*0.577216649

    L = np.matmul(L-cent, np.linalg.pinv([Aah])).flatten()
    
    # boost confidence of primary TCRNet
    confs_overall = confs + L

    row_dets = row_dets/scale_factor
    col_dets = col_dets/scale_factor

    targets = sample['targets']
    nt = len(targets)
    ntgt += nt
    nframes += 1


    for target in targets:
        r = target['center'][1]
        c = target['center'][0]

        # TCRNet-2
        foundtgt = np.zeros(ndets)
        tmpdets = []
        # TCRNet-2  & TCRNet2 booster
        foundtgt2 = np.zeros(ndets)
        tmpdets_overall = []


        for i in range(ndets):
            dist = ((r - row_dets[i]) ** 2 + (c - col_dets[i]) ** 2)**.5

            if dist < 20:
                foundtgt[i] = 1
                tmpdets.append(confs[i])
                tmpdets_overall.append(confs_overall[i])
        if len(tmpdets) >= 1:
            dets.append(max(tmpdets))              # TCRNet-2
            dets2.append(max(tmpdets_overall))     # TCRNet-2  & TCRNet2 booster

        I = np.where(foundtgt == 0)[0]
        for indx_a in confs[I]:
            fas.append(indx_a)                     # TCRNet-2

        for indx_a2 in confs_overall[I]:
            fas2.append(indx_a2)                   # TCRNet-2  & TCRNet2 booster


    # ### Visual plot of detections by TCRNet with ground truth bounding box
    # # warning: saves all images with detections. This can make the process slower.
    # im2_name1 = sample['name'] + '_' + sample['frame'] + "_detections"
    # im = loadmat(imfile)['image']
    # r = sample['targets'][0]['center'][1]
    # c = sample['targets'][0]['center'][0]
    # f, ax = plt.subplots(figsize=(10, 10))
    # ax.imshow(im, cmap='gray')
    # for l in range(min(len(confs),10)):
    #     ax.text(col_dets[l],row_dets[l],"X ", color="yellow", fontsize=15)
    # # Create a Rectangle patch
    # rect = patches.Rectangle((c-40,r-20),80,40,linewidth=1,edgecolor='r',facecolor='none')
    # # Add the patch to the Axes
    # ax.add_patch(rect)
    # # plt.show()
    # plt.savefig("./output/" + im2_name1)


### save outputs to plot ROC curve
# TCRNet-2
dets = np.array(dets)
np.save('./output/dets',dets)
fas = np.array(fas)
np.save('./output/fas',fas)
ntgt = np.array([ntgt])
np.save('./output/ntgt',ntgt)
nframes = np.array([nframes])
np.save('./output/nframes',nframes)

# TCRNet 2 & TCRNet2 booster [/Boosted TCRNet-2]
dets2 = np.array(dets2)
np.save('./output/dets2',dets2)
fas2 = np.array(fas2)
np.save('./output/fas2',fas2)

# print('TCRNet-2 :', ntgt,nframes,len(dets),len(fas))
# print('TCRNet-2  & TCRNet2 booster:', ntgt,nframes,len(dets2),len(fas2))
