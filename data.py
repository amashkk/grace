"""
Data loading utilities for GRACE training and evaluation.

Supports:
- CIFAR-100 (training / ID evaluation, auto-downloaded via torchvision)
- CIFAR-100-C (corruption-based OOD evaluation)

Note: CIFAR-100 images are 32×32; they are resized to 224×224 for CLIP ViT-B/32.
We do NOT normalize in transforms because PGD operates in [0,1] pixel space.
CLIP's own preprocessing handles normalization internally.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


# CIFAR-100 fine-grained class names (all 100 classes)
CIFAR100_CLASSNAMES = [
    "apple", "aquarium fish", "baby", "bear", "beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak tree", "orange", "orchid",
    "otter", "palm tree", "pear", "pickup truck", "pine tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow tree", "wolf", "woman", "worm",
]


def get_train_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.RandomCrop(image_size, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])


def get_eval_transform(image_size: int = 224):
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])


def build_cifar100_loaders(
    root: str,
    batch_size: int,
    num_workers: int = 4,
    image_size: int = 224,
):
    """Build CIFAR-100 train and test dataloaders. Auto-downloads if needed."""
    train_dataset = datasets.CIFAR100(
        root=root, train=True, download=True,
        transform=get_train_transform(image_size),
    )
    test_dataset = datasets.CIFAR100(
        root=root, train=False, download=True,
        transform=get_eval_transform(image_size),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, test_loader, CIFAR100_CLASSNAMES


class CIFAR100C(Dataset):
    """
    CIFAR-100-C corruption benchmark dataset.
    Download from: https://zenodo.org/record/3555552
    Place at: root/CIFAR-100-C/<corruption>.npy and root/CIFAR-100-C/labels.npy
    """

    def __init__(self, root: str, corruption: str, severity: int = 5,
                 transform=None):
        self.transform = transform
        data_path = os.path.join(root, "CIFAR-100-C", f"{corruption}.npy")
        label_path = os.path.join(root, "CIFAR-100-C", "labels.npy")

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"CIFAR-100-C not found at {data_path}. "
                "Download from https://zenodo.org/record/3555552"
            )

        all_data = np.load(data_path)
        all_labels = np.load(label_path)

        start = (severity - 1) * 10000
        end = severity * 10000
        self.data = all_data[start:end]
        self.labels = all_labels[start:end]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = Image.fromarray(self.data[idx])
        label = int(self.labels[idx])
        if self.transform:
            img = self.transform(img)
        return img, label


CIFAR100C_CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness",
    "contrast", "elastic_transform", "pixelate", "jpeg_compression",
]


def build_cifar100c_loader(
    root: str, corruption: str, severity: int = 5,
    batch_size: int = 64, num_workers: int = 4, image_size: int = 224,
):
    dataset = CIFAR100C(
        root=root, corruption=corruption, severity=severity,
        transform=get_eval_transform(image_size),
    )
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
