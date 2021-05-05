import torch
import torch.nn as nn


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.cb = Module_CharbonnierLoss()
        self.l1 = nn.SmoothL1Loss()
    
    def forward(self, output, gt):
        loss_cb = self.cb(output, gt)
        
        out_row_diff = self.l1(output[:,:,1:,:], output[:,:,:-1,:])
        out_col_diff = self.l1(output[:,:,:,1:], output[:,:,:,:-1])
        gt_row_diff = self.l1(gt[:,:,1:,:], gt[:,:,:-1,:])
        gt_col_diff = self.l1(gt[:,:,:,1:], gt[:,:,:,:-1])

        loss_gdl = self.l1(out_row_diff, gt_row_diff) + \
                   self.l1(out_col_diff, gt_col_diff)

        return loss_cb + loss_gdl


class Module_CharbonnierLoss(nn.Module):
    def __init__(self, epsilon=0.001):
        super(Module_CharbonnierLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, output, gt):
        return torch.mean(torch.sqrt((output - gt) ** 2 + self.epsilon ** 2))