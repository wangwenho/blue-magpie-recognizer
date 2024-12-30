import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

class DataWrapper(torch.utils.data.Dataset):
    """
    Apply transformations to the dataset
    """
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def get_label_map(dataset_dir: str = 'dataset_1500'):
    """
    Get the label map of the dataset
    """

    class_map = datasets.ImageFolder(root=dataset_dir).class_to_idx
    label_map = {v: k for k, v in class_map.items()}

    return label_map

def split_dataset(
        dataset_dir: str,
        train_transform,
        test_transform,
        split_ratio: list = [0.7, 0.15, 0.15],
        random_seed: int = 2024,
    ):
    """
    Split the dataset into train, validation, and test sets
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

    train_ratio, val_ratio, test_ratio = split_ratio
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))

    torch.manual_seed(random_seed)

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_dataset = DataWrapper(train_dataset, transform=train_transform)
    val_dataset = DataWrapper(val_dataset, transform=test_transform)
    test_dataset = DataWrapper(test_dataset, transform=test_transform)

    return train_dataset, val_dataset, test_dataset

def get_mean_std(
        dataset_dir: str = 'dataset_1500',
        split_ratio: list = [0.7, 0.15, 0.15],
        random_seed: int = 2024,
    ):
    """
    Calculate the mean and std of the train dataset
    """

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder(root=dataset_dir, transform=transform)

    train_ratio, val_ratio, test_ratio = split_ratio
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = int(test_ratio * len(dataset))
    
    torch.manual_seed(random_seed)

    train_dataset, _, _ = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    dataset_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    mean = 0.0
    std = 0.0

    for images, _ in tqdm(dataset_loader, desc="Calculating mean and std"):
        mean += images.mean([0, 2, 3])
        std += images.std([0, 2, 3])

    mean /= len(dataset_loader)
    std /= len(dataset_loader)

    return mean, std