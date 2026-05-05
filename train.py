import os
import sys

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.functional import structural_similarity_index_measure
from model import FDCFormer
from losses import CombinedLoss
from dataloader import create_dataloaders
import lpips
import math
from torch.optim.lr_scheduler import LambdaLR
# from basicsr.metrics.niqe_metric import calculate_niqe


def build_scheduler(optimizer, num_epochs, warmup_epochs, base_lr, eta_min=1e-6):
    eta_min_ratio = eta_min / base_lr
    total_cosine_epochs = num_epochs - warmup_epochs

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # linear warmup
            return float(epoch + 1) / float(warmup_epochs)
        else:
            # cosine decay
            t = float(epoch - warmup_epochs) / float(max(1, total_cosine_epochs))
            cosine = 0.5 * (1.0 + math.cos(math.pi * t))
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine

    return LambdaLR(optimizer, lr_lambda=lr_lambda)


def calculate_psnr(img1, img2, max_pixel_value=1.0, gt_mean=False):
    """
    Calculate PSNR (Peak Signal-to-Noise Ratio) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The PSNR value.
    """
    if gt_mean:
        img1_gray = img1.mean(axis=1)
        img2_gray = img2.mean(axis=1)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    mse = F.mse_loss(img1, img2, reduction='mean')
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_pixel_value / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(img1, img2, max_pixel_value=1.0, gt_mean=False):
    """
    Calculate SSIM (Structural Similarity Index) between two images.

    Args:
        img1 (torch.Tensor): First image (BxCxHxW)
        img2 (torch.Tensor): Second image (BxCxHxW)
        max_pixel_value (float): The maximum possible pixel value of the images. Default is 1.0.

    Returns:
        float: The SSIM value.
    """
    if gt_mean:
        img1_gray = img1.mean(axis=1, keepdim=True)
        img2_gray = img2.mean(axis=1, keepdim=True)

        mean_restored = img1_gray.mean()
        mean_target = img2_gray.mean()
        img1 = torch.clamp(img1 * (mean_target / mean_restored), 0, 1)

    ssim_val = structural_similarity_index_measure(img1, img2, data_range=max_pixel_value)
    return ssim_val.item()


def validate(model, dataloader, device, lpips_model):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    total_lpips = 0

    with torch.no_grad():
        for low, high in dataloader:
            low, high = low.to(device), high.to(device)
            output = model(low)

            # Calculate PSNR
            psnr = calculate_psnr(output, high)
            total_psnr += psnr

            # Calculate SSIM
            ssim = calculate_ssim(output, high)
            total_ssim += ssim

            # Calculate LPIPS
            output_normalized = torch.clamp(output, 0, 1)
            high_normalized = torch.clamp(high, 0, 1)
            output_lpips = 2 * output_normalized - 1
            high_lpips = 2 * high_normalized - 1

            lpips_value = lpips_model(output_lpips, high_lpips)
            total_lpips += lpips_value.mean().item()

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    avg_lpips = total_lpips / len(dataloader)

    return avg_psnr, avg_ssim, avg_lpips


def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.detach().data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a") 

    def write(self, message):
        self.terminal.write(message)    
        self.log.write(message)         
        self.log.flush()               

    def flush(self):
        self.log.flush()


def main():
    sys.stdout = Logger("training_log7.txt")
    print(f"Log file created at: {os.path.abspath('training_log7.txt')}")
    
    train_low = 'data/LOLv1/Train/input'
    train_high = 'data/LOLv1/Train/target'
    test_low = 'data/LOLv1/Test/input'
    test_high = 'data/LOLv1/Test/target'
   
    learning_rate = 2e-4
    num_epochs = 3000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    resume_path = 'latest_checkpoint6.pth'  # 恢复训练
    print(f'LR: {learning_rate}; Epochs: {num_epochs}')

    # Data loaders
    train_loader, test_loader = create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256,
                                                   batch_size=8)
    print(f'Train loader: {len(train_loader)}; Test loader: {len(test_loader)}')

    # Model, loss, optimizer, and scheduler
    model = FDCFormer().to(device)
    # if torch.cuda.device_count() > 1:
    #     model = torch.nn.DataParallel(model)

    criterion = CombinedLoss(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    # scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)  # 余弦退火策略
    warmup_epochs = 10
    scheduler = build_scheduler(optimizer, num_epochs=num_epochs, warmup_epochs=warmup_epochs,
                                base_lr=learning_rate, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler()

    lpips_model_val = lpips.LPIPS(net='alex').to(device)

    start_epoch = 0
    best_psnr = 0
    # if resume_path and os.path.isfile(resume_path):
    #     print(f"Loading checkpoint '{resume_path}'...")
    #     checkpoint = torch.load(resume_path, map_location=device)
    #
    #     # 加载所有状态
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    #     start_epoch = checkpoint['epoch'] + 1
    #     best_psnr = checkpoint['best_psnr']
    #     print(f"Loaded checkpoint '{resume_path}' (resuming from epoch {start_epoch})")
    # else:
    #     print("No checkpoint found. Training from scratch.")
    print('Training started.')
    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        for batch_idx, batch in enumerate(train_loader):
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()

            # with torch.cuda.amp.autocast():
            #     outputs = model(inputs)
            #     loss = criterion(outputs, targets)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)  # 增加的，缩回真实梯度，用于裁剪

            # if batch_idx % 30 == 0:
            #     grad_norm = get_grad_norm(model)
            #     print(f"Epoch {epoch + 1}, Batch {batch_idx}, Loss: {loss.item():.6f}, Grad Norm: {grad_norm:.4f}")

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)  # 原始max_norm=5.0
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        avg_psnr, avg_ssim, avg_lpips = validate(model, test_loader, device, lpips_model_val)
        avg_loss = train_loss / len(train_loader)
        # print(f'Epoch {epoch + 1}/{num_epochs}, PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}, LPIPS: {avg_lpips:.6f}, '
        #       f'Loss: {avg_loss:.6f}')

        # 打印当前 lr（可选）
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f'Epoch {epoch + 1}/{num_epochs}, PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}, '
            f'LPIPS: {avg_lpips:.6f}, Loss: {avg_loss:.6f}, LR: {current_lr:.6e}'
            )
        scheduler.step()

        checkpoint_state = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'best_psnr': best_psnr
        }
        save_checkpoint(checkpoint_state, filename='latest_checkpoint.pth')

        if epoch % 30 == 0 and epoch > 100:
            # torch.save(model.state_dict(), f'modelv12.17.15_Huawei_Batch2_1000eps_lr0.0001_{epoch}.pth')
            torch.save(model.state_dict(), f'modelv4.1_Huawei_Batch8_500eps_lr0.0002_{epoch}.pth')

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            # torch.save(model.state_dict(), 'modelv12.17.15_Huawei_Batch8_1000eps_lr0.0002.pth')
            torch.save(model.state_dict(), 'modelv4.1_Huawei_Batch8_500eps_lr0.0002.pth')
            print(f'Saving model with PSNR: {best_psnr:.6f}')


if __name__ == '__main__':
    main()
