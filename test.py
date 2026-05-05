import torch
import torch.nn.functional as F
from torchmetrics.functional import structural_similarity_index_measure
# from modelv12 import FDCFormer
from modelv4 import FDCFormer
from dataloader import create_dataloaders, create_unpaired_test_loader
import os
import numpy as np
from torchvision.utils import save_image


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


def validate(model, dataloader, device, result_dir):
    model.eval()
    total_psnr = 0
    total_ssim = 0
    with torch.no_grad():
        for idx, (low, high) in enumerate(dataloader):
            low, high = low.to(device), high.to(device)
            output = model(low)
            output = torch.clamp(output, 0, 1)

            # Save the output image
            save_image(output, os.path.join(result_dir, f'result_{idx}.png'))
            # save_image(output, os.path.join(result_dir, f'result_{idx}.jpg'))

            # Calculate PSNR
            psnr = calculate_psnr(output, high)
            total_psnr += psnr

            # Calculate SSIM
            ssim = calculate_ssim(output, high)
            total_ssim += ssim

    avg_psnr = total_psnr / len(dataloader)
    avg_ssim = total_ssim / len(dataloader)
    return avg_psnr, avg_ssim


def validate_unpaired(model, dataloader, device, result_dir):
    model.eval()

    with torch.no_grad():
        for idx, (low, name) in enumerate(dataloader):
            low = low.to(device)

            # --- 自动补齐到 8 的倍数 ---
            _, _, h0, w0 = low.size()
            # pad_h = (8 - h0 % 8) % 8
            # pad_w = (8 - w0 % 8) % 8
            pad_h = (16 - h0 % 16) % 16
            pad_w = (16 - w0 % 16) % 16

            low_padded = F.pad(low, (0, pad_w, 0, pad_h), mode='reflect')

            # --- 推理 ---
            output = model(low_padded)

            # --- 裁掉 padding ---
            output = output[:, :, :h0, :w0]

            output = torch.clamp(output, 0, 1)

            save_image(output, os.path.join(result_dir, f'enhanced_{name[0]}'))


def main():
    # Paths and device setup
    # test_low = 'data/LOLv1/Test/input'
    # test_high = 'data/LOLv1/Test/target'
    # test_low = 'data/LOLv2/Real_captured/Test/Low'
    # test_high = 'data/LOLv2/Real_captured/Test/Normal'
    # test_low = 'data/LOLv2/Synthetic/Test/Low'
    # test_high = 'data/LOLv2/Synthetic/Test/Normal'
    test_low = 'data/Huawei/Train/low'
    test_high = 'data/Huawei/Train/high'
    # weights_path = 'best_model_LOLv1.pth'
    # weights_path = 'best_model.pth'
    # weights_path = 'WNENet_best_model.pth'
    # weights_path = 'modelv7_FDCB_LOLv2sys_noGT_Batch2_300eps_lr0.0002.pth'
    # weights_path = 'modelv5_topk_LOLv1_noGT_Batch8_1000eps_lr0.0002.pth'
    # weights_path = 'modelv5_topk_LOLv2real_noGT_Batch2_500eps_lr0.0002.pth'
    # weights_path = 'modelv12.14_LOLv2_real_noGT_Batch16_1000eps_lr0.0002.pth'
    # weights_path = 'modelv12.14_LOLv2_real_noGT_Batch12_1000eps_lr0.0004.pth'
    # weights_path = 'modelv12.14_LOLv2_real_noGT_Batch12_500eps_lr0.0004.pth'
    # weights_path = 'modelv12.15_LOLv2_real_noGT_Batch12_500eps_lr0.0004.pth'
    # weights_path = 'modelv12.16_LOLv2_real_noGT_Batch12_500eps_lr0.001.pth'
    # weights_path = 'modelv12.16.1_LOLv2_real_noGT_Batch12_500eps_lr0.001.pth'
    # weights_path = 'modelv12.17_LOLv2_real_noGT_Batch12_500eps_lr0.001.pth'
    # weights_path = 'modelv12.17.3_LOLv1_noGT_Batch8_500eps_lr0.0004.pth'
    # weights_path = 'modelv12.17.13_LOLv2_real_noGT_Batch8_1000eps_lr0.0004.pth'
    # weights_path = 'modelv12.17.14_LOLv2_real_noGT_Batch2_1000eps_lr0.0004.pth'
    # weights_path = 'modelv12.17.14_LOLv1_noGT_Batch2_1000eps_lr0.0004.pth'
    # weights_path = 'modelv12.17.14.1_LOLv1_noGT_Batch8_1000eps_lr0.0002.pth'
    # weights_path = 'modelv12.17.14.3_LOLv1_noGT_Batch2_2000eps_lr0.0004.pth'
    # weights_path = 'modelv12.17.14.3_LOLv1_noGT_Batch2_2000eps_lr0.0006.pth'
    # weights_path = 'modelv12.17.14.3_LOLv1_noGT_Batch2_4000eps_lr0.0005.pth'
    # weights_path = 'modelv12.17.14.4_LOLv1_noGT_Batch2_2000eps_lr0.0004.pth'
    # weights_path = 'modelv12.17.14.5_LOLv1_noGT_Batch2_2000eps_lr0.0004.pth'
    # weights_path = 'modelv12.17.14.5_LOLv1_noGT_Batch2_2000eps_lr0.0002.pth'
    # weights_path = 'modelv12.17.15_LOLv1_noGT_Batch8_1000eps_lr0.0001.pth'
    # weights_path = 'modelv12.17.15_LOLv2_s_Batch2_1000eps_lr0.0002.pth'
    # weights_path = 'modelv12.17.15_LOLv2_s_Batch2_1000eps_lr0.0002_950.pth'
    # weights_path = 'modelv7_FDCB.3_LOLv2_sys_noGT_Batch8_100eps_lr0.0002.pth'
    # weights_path = 'modelv12.17.15_Huawei_Batch8_500eps_lr0.00001.pth'
    weights_path = 'modelv4.1_Huawei_Batch8_500eps_lr0.0002.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset_name = test_low.split('/')[1]
    # result_dir = os.path.join('results_modelv7', dataset_name)
    # result_dir = os.path.join('results_modelv12.17.15.Huawei1', dataset_name)
    result_dir = os.path.join('results_modelv4.1.Huawei', dataset_name)
    os.makedirs(result_dir, exist_ok=True)

    _, test_loader = create_dataloaders(None, None, test_low, test_high, crop_size=None, batch_size=1)
    print(f'Test loader: {len(test_loader)}')

    model = FDCFormer().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f'Model loaded from {weights_path}')

    avg_psnr, avg_ssim = validate(model, test_loader, device, result_dir)
    print(f'Validation PSNR: {avg_psnr:.6f}, SSIM: {avg_ssim:.6f}')


def main1():
    test_low = 'data/NPE'   # 非配对
    # weights_path = 'modelv12.17.15_LOLv2_s_Batch2_1000eps_lr0.0002.pth'
    # weights_path = 'modelv12.17.15_LOLv2_s_Batch2_1000eps_lr0.0002_950.pth'
    # weights_path = 'FDCFv4_LOLv1_noGT_Batch8_500eps_lr0.0002.pth'
    # weights_path = 'FDCFv4_LOLv1_noGT_Batch8_100eps_lr0.0002.pth'
    # weights_path = 'modelv9_LOLv1_noGT_Batch8_4000eps_lr0.0002.pth'
    # weights_path = 'modelv8_LOLv2_noGT_Batch8_500eps_lr0.0002.pth'
    # weights_path = 'modelv7_FDCB.3_LOLv2_sys_noGT_Batch8_100eps_lr0.0002.pth'
    weights_path = 'modelv7_FDCB.3_LOLv2sys_noGT_Batch2_300eps_lr0.0002.pth'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    result_dir = 'results_unpaired_21'
    os.makedirs(result_dir, exist_ok=True)

    # --- Unpaired test loader ---
    test_loader = create_unpaired_test_loader(test_low, batch_size=1)
    print(f'Unpaired test samples: {len(test_loader)}')

    # --- Load model ---
    model = FDCFormer().to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    # --- Run unpaired validation ---
    validate_unpaired(model, test_loader, device, result_dir)
    print("Done. Enhanced results saved in:", result_dir)


if __name__ == '__main__':
    main()
