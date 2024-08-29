import torch.utils.data
from torch import nn


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        logits = self.sigmoid(inputs)
        TP = (logits * targets).sum(dim=(2, 3, 4))
        total = (targets + logits).sum(dim=(2, 3, 4))
        return (1 - ((2 * TP + 1) / (total + 1))).mean()


class TverskyLoss(nn.Module):
    def __init__(self, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.beta = beta
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        logits = self.sigmoid(inputs)
        TP = (logits * targets).sum(dim=(2, 3, 4))
        FN = (targets * (1 - logits)).sum(dim=(2, 3, 4))
        FP = ((1 - targets) * logits).sum(dim=(2, 3, 4))
        return (1 - (TP / (TP + (1 - self.beta) * FN + self.beta * FP))).mean()


class FocalTverskyLoss(nn.Module):
    def __init__(self, beta=0.5, gamma=2):
        super(FocalTverskyLoss, self).__init__()
        self.beta = beta
        self.gamma = gamma
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs, targets):
        logits = self.sigmoid(inputs)
        TP = (logits * targets).sum(dim=(2, 3, 4))
        FN = (targets * (1 - logits)).sum(dim=(2, 3, 4))
        FP = ((1 - targets) * logits).sum(dim=(2, 3, 4))
        TI = TP / (TP + (1 - self.beta) * FN + self.beta * FP)
        return torch.pow(1 - TI, self.gamma).mean()


# TODO: implement SegNet
# TODO: implement FCN
class SDFUnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv3 = nn.Conv3d(in_channels=8, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv4 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv5 = nn.Conv3d(in_channels=16, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv6 = nn.Conv3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv7 = nn.Conv3d(in_channels=48, out_channels=16, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv8 = nn.Conv3d(in_channels=24, out_channels=8, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.conv9 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=(3, 3, 3), stride=(1, 1, 1),
                               padding=(1, 1, 1))
        self.deconv1 = nn.ConvTranspose3d(in_channels=32, out_channels=32, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                          padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.deconv2 = nn.ConvTranspose3d(in_channels=16, out_channels=16, kernel_size=(3, 3, 3), stride=(2, 2, 2),
                                          padding=(1, 1, 1), output_padding=(1, 1, 1))
        self.sigmoid = nn.Sigmoid()
        self.ReLU = nn.ReLU()
        self.max_pool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1), return_indices=True)
        self.batch_norm_1 = nn.BatchNorm3d(8)
        self.batch_norm_2 = nn.BatchNorm3d(16)
        self.batch_norm_3 = nn.BatchNorm3d(32)

    def forward(self, x):
        # TODO: modify to https://www.researchgate.net/profile/Olaf-Ronneberger/publication/304226155_3D_U-Net_Learning_Dense_Volumetric_Segmentation_from_Sparse_Annotation/links/5b2a4c990f7e9b1d009c9b60/3D-U-Net-Learning-Dense-Volumetric-Segmentation-from-Sparse-Annotation.pdf
        logits = self.conv1(x)
        logits = self.batch_norm_1(logits)
        logits = self.ReLU(logits)
        logits = self.conv2(logits)
        logits = self.batch_norm_1(logits)
        logits = self.ReLU(logits)
        skip1 = torch.clone(logits)
        logits, _ = self.max_pool(logits)
        logits = self.conv3(logits)
        logits = self.batch_norm_2(logits)
        logits = self.ReLU(logits)
        logits = self.conv4(logits)
        logits = self.batch_norm_2(logits)
        logits = self.ReLU(logits)
        skip2 = torch.clone(logits)
        logits, _ = self.max_pool(logits)
        logits = self.conv5(logits)
        logits = self.batch_norm_3(logits)
        logits = self.ReLU(logits)
        logits = self.conv6(logits)
        logits = self.batch_norm_3(logits)
        logits = self.ReLU(logits)
        logits = self.conv6(logits)
        logits = self.batch_norm_3(logits)
        logits = self.ReLU(logits)
        logits = self.deconv1(logits)
        logits = torch.cat((logits, skip2), dim=1)
        logits = self.conv7(logits)
        logits = self.batch_norm_2(logits)
        logits = self.ReLU(logits)
        logits = self.deconv2(logits)
        logits = torch.cat((logits, skip1), dim=1)
        logits = self.conv8(logits)
        logits = self.batch_norm_1(logits)
        logits = self.ReLU(logits)
        logits = self.conv9(logits)
        return logits
