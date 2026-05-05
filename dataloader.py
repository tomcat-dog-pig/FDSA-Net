import os

import torch
from PIL import Image
import glob
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random


class PairedDataset(Dataset):
    def __init__(self, low_dir, high_dir, transform=None, crop_size=None, training=True):
        self.low_dir = low_dir
        self.high_dir = high_dir
        self.transform = transform
        self.crop_size = crop_size
        self.training = training

        self.low_images = sorted([f for f in os.listdir(low_dir) if os.path.isfile(os.path.join(low_dir, f))])
        self.high_images = sorted([f for f in os.listdir(high_dir) if os.path.isfile(os.path.join(high_dir, f))])

        assert len(self.low_images) == len(self.high_images), "Mismatch in number of images"

    def __len__(self):
        return len(self.low_images)

    def __getitem__(self, idx):
        low_image_path = os.path.join(self.low_dir, self.low_images[idx])
        high_image_path = os.path.join(self.high_dir, self.high_images[idx])

        low_image = Image.open(low_image_path).convert('RGB')
        high_image = Image.open(high_image_path).convert('RGB')

        if self.transform:
            low_image = self.transform(low_image)
            high_image = self.transform(high_image)

        if self.crop_size:
            i, j, h, w = transforms.RandomCrop.get_params(low_image, output_size=(self.crop_size, self.crop_size))
            low_image = transforms.functional.crop(low_image, i, j, h, w)
            high_image = transforms.functional.crop(high_image, i, j, h, w)

        if self.training:
            aug = random.randint(0, 8)
            if aug == 1:
                low_image = low_image.flip(1)
                high_image = high_image.flip(1)
            elif aug == 2:
                low_image = low_image.flip(2)
                high_image = high_image.flip(2)
            elif aug == 3:
                low_image = torch.rot90(low_image, dims=(1, 2))
                high_image = torch.rot90(high_image, dims=(1, 2))
            elif aug == 4:
                low_image = torch.rot90(low_image, dims=(1, 2), k=2)
                high_image = torch.rot90(high_image, dims=(1, 2), k=2)
            elif aug == 5:
                low_image = torch.rot90(low_image, dims=(1, 2), k=3)
                high_image = torch.rot90(high_image, dims=(1, 2), k=3)
            elif aug == 6:
                low_image = torch.rot90(low_image.flip(1), dims=(1, 2))
                high_image = torch.rot90(high_image.flip(1), dims=(1, 2))
            elif aug == 7:
                low_image = torch.rot90(low_image.flip(2), dims=(1, 2))
                high_image = torch.rot90(high_image.flip(2), dims=(1, 2))

        return low_image, high_image


def create_dataloaders(train_low, train_high, test_low, test_high, crop_size=256, batch_size=8):
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    train_loader = None
    test_loader = None

    if train_low and train_high:
        train_dataset = PairedDataset(train_low, train_high, transform=transform, crop_size=crop_size, training=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    if test_low and test_high:
        test_dataset = PairedDataset(test_low, test_high, transform=transform, crop_size=None, training=False)
        test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    return train_loader, test_loader


class UnpairedTestDataset(Dataset):
    def __init__(self, low_dir):
        self.files = sorted(glob.glob(os.path.join(low_dir, '*')))
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)
        return img, os.path.basename(img_path)


def create_unpaired_test_loader(low_dir, batch_size=1):
    dataset = UnpairedTestDataset(low_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

