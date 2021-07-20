import torch.nn as nn
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import scipy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class tcrNet(nn.Module):
    def __init__(self):
        super(tcrNet, self).__init__()
        self.cnv_tgt = nn.Conv2d(in_channels=1, out_channels=70, kernel_size=(20,40), stride=1, padding=0)
        self.tgt_bn1 = nn.BatchNorm2d(70)
        self.cnv_tgt2 = nn.Conv2d(in_channels=70, out_channels=50, kernel_size=3, stride=1, padding=0)
        self.tgt_bn2 = nn.BatchNorm2d(50)
        self.cnv_tgt3 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=0)

        self.cnv_clt = nn.Conv2d(in_channels=1, out_channels=30, kernel_size=(20,40), stride=1, padding=0)
        self.clt_bn1 = nn.BatchNorm2d(30)
        self.cnv_clt2 = nn.Conv2d(in_channels=30, out_channels=50, kernel_size=3, stride=1, padding=0)
        self.clt_bn2 = nn.BatchNorm2d(50)
        self.cnv_clt3 = nn.Conv2d(in_channels=50, out_channels=50, kernel_size=3, stride=1, padding=0)
        
        self.bn = nn.BatchNorm2d(100)
        self.cnv4 = nn.Conv2d(in_channels=100, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.leakyrelu1 = nn.LeakyReLU(0.1)
        self.leakyrelu2 = nn.LeakyReLU(0.01)

        # engineered filters for first layer
        target_filters = np.load('./weights_filters/target_filters.npy')
        clutter_filters = np.load('./weights_filters/clutter_filters.npy')

        print('targets filters:', target_filters.shape)
        print('clutter filters:', clutter_filters.shape)
        
        target_filters = target_filters[:, 0:70]
        target_filters = np.swapaxes(target_filters, 0, 1)
        target_filters = target_filters.reshape((-1, 40, 20))
        target_filters = np.expand_dims(target_filters, axis=1)
        target_filters = np.swapaxes(target_filters, 2, 3)

        clutter_filters = clutter_filters[:, 0:30]
        clutter_filters = np.swapaxes(clutter_filters, 0, 1)
        clutter_filters = clutter_filters.reshape((-1, 40, 20))
        clutter_filters = np.expand_dims(clutter_filters, axis=1)
        clutter_filters = np.swapaxes(clutter_filters, 2, 3)
        
        print('targets', target_filters.shape)
        print('clutter', clutter_filters.shape)

        layer_tgt = torch.tensor(target_filters).float()
        self.cnv_tgt.weight = nn.Parameter(layer_tgt)
        self.cnv_tgt.weight.requires_grad = False

        layer_clt = torch.tensor(clutter_filters).float()
        self.cnv_clt.weight = nn.Parameter(layer_clt)
        self.cnv_clt.weight.requires_grad = False

    def forward(self, x):
        x_tgt = self.leakyrelu1(self.tgt_bn1(self.cnv_tgt(x)))
        x_tgt = self.leakyrelu1(self.tgt_bn2(self.cnv_tgt2(x_tgt)))
        x_tgt = self.cnv_tgt3(x_tgt)

        x_clt = self.leakyrelu1(self.clt_bn1(self.cnv_clt(x)))
        x_clt = self.leakyrelu1(self.clt_bn2(self.cnv_clt2(x_clt)))
        x_clt = self.cnv_clt3(x_clt)

        x_combine = torch.cat((x_clt, x_tgt), dim=1)
        x_combine = self.leakyrelu2(self.bn(x_combine))
        x_combine = self.cnv4(x_combine)

        return x_combine

class tcrLoss(torch.autograd.Function):

    # Note that both forward and backward are @staticmethods
    @staticmethod
    def forward(ctx, predictions,gt):
        ctx.save_for_backward(predictions, gt)
        sum = torch.sum(gt,dim=3)
        sum = torch.sum(sum,dim=2)
        clutter_idx = torch.where(sum == 0)[0]
        target_idx = torch.where(sum != 0)[0]

        # print('clutter',clutter_idx,'targets',target_idx)

        target_response = predictions[target_idx,:,:,:].squeeze()
        clutter_response = predictions[clutter_idx,:,:,:].squeeze()

        target_response = target_response **2
        clutter_response = clutter_response **2

        target_peak = target_response[:,8,18]  #corresponds to gaussian peak in gt, detect instead of hard code later
        clutter_energy = torch.sum(clutter_response,dim=2)
        clutter_energy = torch.sum(clutter_energy,dim=1)

        # print('peak',target_peak.shape,'clutter',clutter_energy)
        n1 = target_peak.shape[0]
        n2 = clutter_energy.shape[0]
        if n1 != 0:
            loss1 = torch.log(target_peak.sum()/n1)
            # loss1 = torch.log(target_peak).sum()/n1

        else:
            loss1 = 0

        if n2 != 0:
            loss2 = torch.log(clutter_energy.sum()/n2)
            # loss2 = torch.log(clutter_energy.sum())

        else:
            loss2 = 0

        loss = loss2 - loss1

        return loss

    # This function has only a single output, so it gets only one gradient
    @staticmethod
    def backward(ctx, grad_output):

        predictions, gt = ctx.saved_tensors
        grad_gt = None
        grad_input = None

        sum = torch.sum(gt, dim=3)
        sum = torch.sum(sum, dim=2)

        clutter_idx = torch.where(sum == 0)[0]
        target_idx = torch.where(sum != 0)[0]

        n_samples = predictions.shape[0]

        category = torch.zeros(n_samples)
        category[target_idx] = 1


        # print('clutter',clutter_idx,'targets',target_idx)

        target_response = predictions[target_idx, :, :, :].squeeze()
        clutter_response = predictions[clutter_idx, :, :, :].squeeze()

        n_targets = target_response.shape[0]
        n_clutter = clutter_response.shape[0]

        U = torch.zeros((17,37)).to(device)
        U[8,18] = 1

        target_peak_energy = torch.zeros(n_targets)
        target_offpeak_energy = torch.zeros(n_targets)
        for i in range (n_targets):
            tmp = U * target_response[i]
            tmp = torch.flatten(tmp).unsqueeze(0)
            target_peak_energy[i] = torch.mm(tmp, torch.transpose(tmp,0,1))

            tmp = (1 - U) * target_response[i]
            tmp = torch.flatten(tmp).unsqueeze(0)
            target_offpeak_energy[i] = torch.mm(tmp, torch.transpose(tmp, 0, 1))

        clutter_response = clutter_response ** 2

        clutter_energy = torch.sum(clutter_response, dim=2)
        clutter_energy = torch.sum(clutter_energy, dim=1)

        idx_clutter = 0
        idx_target = 0
        grad_input = predictions.clone()

        for i in range(n_samples):
            if category[i] == 0:
                grad_input[i,0,:,:] = predictions[i,0,:,:]/clutter_energy[idx_clutter]
                idx_clutter += 1
            else:
                tmp = predictions[i,0,:,:]
                UY = U * tmp
                BY = (1-U)* tmp
                grad_input[i, 0, :, :] = BY/target_offpeak_energy[idx_target] -UY/(target_peak_energy[idx_target] + 1e-5)
                idx_target += 1

        return grad_input, grad_gt