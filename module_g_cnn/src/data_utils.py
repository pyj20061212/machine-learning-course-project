import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def get_mnist_dataloaders(
    data_dir: str = "../data",
    batch_size: int = 64,
    val_size: int = 5000,
    num_workers: int = 0,
    augment: bool = False,
    normalize: bool = False,
    seed: int = 42,
):
    """
    Return train/val/test dataloaders for MNIST.

    Parameters
    ----------
    data_dir : str
        Directory to store/download MNIST.
    batch_size : int
        Batch size for dataloaders.
    val_size : int
        Validation set size split from official train set.
    num_workers : int
        Number of workers for DataLoader.
    augment : bool
        Whether to apply light augmentation on training set.
    normalize : bool
        Whether to standardize MNIST using mean/std.
    seed : int
        Random seed for deterministic train/val split.
    """
    train_transforms = []
    test_transforms = []

    if augment:
        train_transforms.extend([
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.95, 1.05),
            )
        ])

    train_transforms.append(transforms.ToTensor())
    test_transforms.append(transforms.ToTensor())

    if normalize:
        mean, std = (0.1307,), (0.3081,)
        train_transforms.append(transforms.Normalize(mean, std))
        test_transforms.append(transforms.Normalize(mean, std))

    train_transform = transforms.Compose(train_transforms)
    test_transform = transforms.Compose(test_transforms)

    full_train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=train_transform,
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=test_transform,
    )

    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator,
    )

    # validation should not use augmentation randomness from train transform
    if augment or normalize:
        val_dataset.dataset = datasets.MNIST(
            root=data_dir,
            train=True,
            download=True,
            transform=test_transform,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader


def get_class_names():
    return [str(i) for i in range(10)]