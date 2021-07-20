"""
Trains both TCRNet-2 and TCRNet-2 booster
"""

import glob
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torch
# import tcr_cnn
import tcr_cnn_2streams as tcr_cnn

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def show(im):
    f, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(im)
    plt.show()

class nvesd_cnn_train(Dataset):
    def __init__(self, target_files, clutter_files):
        
        
        gaussian = loadmat('./weights_filters/h.mat')['h']

        d1 = 40
        d2 = 80
        nt = len(target_files)
        nc = len(clutter_files)
        print(nt, 'target chips')
        print(nc, 'clutter chips')
        alltargets = np.zeros((d1, d2, nt))
        for idx, chip in enumerate(target_files):
            chiparray = loadmat(chip)['target_chip']
            chiparray = chiparray - chiparray.mean()
            alltargets[:, :, idx] = chiparray

        allclutter = np.zeros((d1, d2, nc))
        for idx, chip in enumerate(clutter_files):
            chiparray = loadmat(chip)['clutter_chip']
            chiparray = chiparray - chiparray.mean()
            allclutter[:, :, idx] = chiparray

        allclutter = np.concatenate((allclutter, allclutter), axis=2)

        print('clutter',allclutter.shape)


        yt = np.tile(gaussian,(nt,1,1))
        print('yt',yt.shape)

        yc = np.tile(np.zeros((17,37)),(nc*2,1,1))
        print('yc',yc.shape)

        self.x = np.concatenate((alltargets,allclutter),axis=2)
        print('x',self.x.shape)

        self.y = np.concatenate((yt,yc),axis=0)
        print('y',self.y.shape)

    def __len__(self):
        return self.x.shape[2]

    def __getitem__(self, idx):
        x = self.x[:,:,idx]
        x = np.expand_dims(x, axis=0)
        y = self.y[idx,:,:]
        y = np.expand_dims(y, axis=0)


        return x,y



def train(target_files, clutter_files, epochs, weight_name):
    net = tcr_cnn.tcrNet().to(device)
    trainset = nvesd_cnn_train(target_files, clutter_files)

    trainloader = DataLoader(
        dataset=trainset,
        batch_size=100,
        num_workers=5,
        shuffle=True
    )
    # results = []

    criterion = tcr_cnn.tcrLoss.apply
    # optimizer = optim.RMSprop(net.parameters(), lr=0.00001, momentum=0.9, weight_decay=.01)
    # optimizer = optim.Adam(net.parameters(), lr=0.00001, weight_decay=.01)
    optimizer = optim.Adam(net.parameters(), lr=0.0001, weight_decay=.002)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=.1)

    losses = []
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        epoch_loss = 0.0
        for i, data in enumerate(trainloader):
            x = data[0].float().to(device)
            gt = data[1].float().to(device)
            optimizer.zero_grad()
            outputs = net(x)
            loss = criterion(outputs, gt)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            epoch_loss += loss.item()
            if i % 10 == 0:    # print every 10 mini-batches
                losses.append(running_loss/10)
                print('[%d, %5d] loss: %.8f' % (epoch, i, running_loss/10))
                running_loss = 0.0
        scheduler.step()
        print('Epoch:', epoch, 'LR:', scheduler.get_last_lr(),'epoch loss',epoch_loss)

    torch.save(net.state_dict(), './weights_filters/' + weight_name +'.pth')
    losses = np.array(losses)
    np.save('./output/losses',losses)
    print('Finished Training')



# Train TCRNet-2
if __name__ == '__main__':
    target_files  = glob.glob('../data/train/chips40x80/targets/' + '*.mat')
    clutter_files = glob.glob('../data/train/chips40x80/clutter/' + '*.mat')
    train(target_files = target_files, clutter_files = clutter_files, epochs = 50, weight_name = 'tcrnet2_weights')
    # losses = np.load('./output/losses.npy')
    # plt.plot(losses)
    # plt.show()


# # Train TCRNet-2 booster
# if __name__ == '__main__':
#     target_files  = glob.glob('../data/train/chips40x80/targets/' + '*.mat')
#     clutter_files = glob.glob('../data/train/chips40x80/hard_clutter/' + '*.mat')
#     train(target_files = target_files, clutter_files = clutter_files, epochs = 50, weight_name = 'tcrnet2_booster_weights')
#     # losses = np.load('./output/losses.npy')
#     # plt.plot(losses)
#     # plt.show()
