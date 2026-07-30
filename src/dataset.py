import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path

# Used by validation and test data
evaluation_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Used by training data
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_path = Path("data") / "train"
test_path = Path("data") / "test"

full_train_dataset = ImageFolder(
    root=train_path,
    transform=train_transform
)

full_validation_dataset = ImageFolder(
    root=train_path,
    transform=evaluation_transform
)

test_dataset = ImageFolder(
    root=test_path,
      transform=evaluation_transform
)

# 80% training, 20% validation
dataset_size = len(full_train_dataset)
train_size = int(0.8 * dataset_size)
validation_size = dataset_size - train_size

# Fixed seed makes the split repeatable
generator = torch.Generator().manual_seed(42)

indices = torch.randperm(
    dataset_size,
    generator=generator
).tolist()

train_indices = indices[:train_size]
validation_indices = indices[train_size:]

train_dataset = Subset(
    full_train_dataset,
    train_indices
)

validation_dataset = Subset(
    full_validation_dataset,
    validation_indices
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

validation_dataloader = DataLoader(
    validation_dataset,
    batch_size=32,
    shuffle=False
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Classes:", full_train_dataset.classes)
print("Class mapping:", full_train_dataset.class_to_idx)

print("Train images:", len(train_dataset))
print("Validation images:", len(validation_dataset))
print("Test images:", len(test_dataset))

images, labels = next(iter(train_dataloader))
print("Image batch shape:", images.shape) # (number of images, channels, height, width)
print("Label batch shape:", labels.shape)

print("Class mapping:", full_train_dataset.class_to_idx)