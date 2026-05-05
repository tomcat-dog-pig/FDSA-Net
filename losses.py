import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from pytorch_msssim import ms_ssim, ssim
import torchvision.transforms as T




def charbonnier_loss(y_true, y_pred, eps=1e-3):
    diff = y_true - y_pred
    return torch.mean(torch.sqrt(diff * diff + eps * eps))


class VGGPerceptualLoss(nn.Module):
    def __init__(self, device):
        super(VGGPerceptualLoss, self).__init__()
        vgg = models.vgg19(weights=True).features[:16]  # Until block3_conv3
        self.loss_model = vgg.to(device).eval()
        for param in self.loss_model.parameters():
            param.requires_grad = False

    def forward(self, y_true, y_pred):
        y_true, y_pred = y_true.to(next(self.loss_model.parameters()).device), y_pred.to(
            next(self.loss_model.parameters()).device)
        return F.mse_loss(self.loss_model(y_true), self.loss_model(y_pred))


def color_loss(y_true, y_pred):
    return torch.mean(torch.abs(torch.mean(y_true, dim=[1, 2, 3]) - torch.mean(y_pred, dim=[1, 2, 3])))


def psnr_loss(y_true, y_pred):
    mse = F.mse_loss(y_true, y_pred)
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    return 40.0 - torch.mean(psnr)


def smooth_l1_loss(y_true, y_pred):
    return F.smooth_l1_loss(y_true, y_pred)


def multiscale_ssim_loss(y_true, y_pred, max_val=1.0):
    # weights = torch.FloatTensor([0.6, 0.3, 0.1]).to(y_true.device)
    return 1.0 - ms_ssim(y_true, y_pred, data_range=max_val, size_average=True)
                         # , win_size=7, weights=weights)


def ssim_loss(y_true, y_pred, max_val=1.0):
    return 1.0 - ssim(y_true, y_pred, data_range=max_val, size_average=True)


def gaussian_kernel(x, mu, sigma):
    return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)


def histogram_loss(y_true, y_pred, bins=256, sigma=0.01):
    bin_edges = torch.linspace(0.0, 1.0, bins, device=y_true.device)

    y_true_hist = torch.sum(gaussian_kernel(y_true.unsqueeze(-1), bin_edges, sigma), dim=0)
    y_pred_hist = torch.sum(gaussian_kernel(y_pred.unsqueeze(-1), bin_edges, sigma), dim=0)

    y_true_hist /= y_true_hist.sum()
    y_pred_hist /= y_pred_hist.sum()

    hist_distance = torch.mean(torch.abs(y_true_hist - y_pred_hist))
    return hist_distance


class FFTLoss(nn.Module):
    """
    L1 loss in frequency domain with FFT.
    Args:
        loss_weight (float): Loss weight for FFT loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output. Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(FFTLoss, self).__init__()
        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (..., C, H, W).Predicted tensor.
            target (Tensor): of shape (..., C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (..., C, H, W). Element-wise weights. Default: None.
        """
        pred_fft = torch.fft.fft2(pred, dim=(-2, -1))
        pred_fft = torch.stack([pred_fft.real, pred_fft.imag], dim=-1)
        target_fft = torch.fft.fft2(target, dim=(-2, -1))
        target_fft = torch.stack([target_fft.real, target_fft.imag], dim=-1)
        return self.loss_weight * F.l1_loss(pred_fft, target_fft, weight, reduction=self.reduction)


class CombinedLoss(nn.Module):
    def __init__(self, device):
        super(CombinedLoss, self).__init__()
        self.perceptual_loss_model = VGGPerceptualLoss(device)
        self.alpha1 = 1.00
        self.alpha2 = 0.06  
        self.alpha3 = 0.5
        # self.alpha4 = 0.5   # 0.5
        # self.alpha5 = 0.0083  # 0.0083
        # self.alpha6 = 0.25

    def forward(self, y_true, y_pred):
        smooth_l1_l = smooth_l1_loss(y_true, y_pred)
        c_loss = charbonnier_loss(y_true, y_pred)
        ms_ssim_l = multiscale_ssim_loss(y_true, y_pred)
        perc_l = self.perceptual_loss_model(y_true, y_pred)
        hist_l = histogram_loss(y_true, y_pred)
        psnr_l = psnr_loss(y_true, y_pred)
        color_l = color_loss(y_true, y_pred)

        total_loss = (
                      self.alpha1 * c_loss
                      + self.alpha2 * perc_l
                      + self.alpha3 * ms_ssim_l
                      )

        return torch.mean(total_loss)
