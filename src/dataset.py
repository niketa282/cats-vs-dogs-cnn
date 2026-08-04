import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from pathlib import Path

# --------------------------------------------------
# Custom CNN transforms: 128 × 128
# --------------------------------------------------

# Used by validation and test data
custom_evaluation_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Used by training data
custom_train_transform  = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
        transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_path = Path("data") / "train"
test_path = Path("data") / "test"

custom_full_train_dataset = ImageFolder(
    root=train_path,
    transform=custom_train_transform
)

full_validation_dataset = ImageFolder(
    root=train_path,
    transform=custom_evaluation_transform
)

test_dataset = ImageFolder(
    root=test_path,
      transform=custom_evaluation_transform
)

# 80% training, 20% validation
dataset_size = len(custom_full_train_dataset)
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
    custom_full_train_dataset,
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

# --------------------------------------------------
# ResNet18 transforms: 224 × 224
# --------------------------------------------------

resnet_evaluation_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

resnet_train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

resnet_full_train_dataset = ImageFolder(
    root=train_path,
    transform=resnet_train_transform
)

resnet_full_validation_dataset = ImageFolder(
    root=train_path,
    transform=resnet_evaluation_transform
)

resnet_test_dataset = ImageFolder(
    root=test_path,
    transform=resnet_evaluation_transform
)

resnet_train_dataset = Subset(
    resnet_full_train_dataset,
    train_indices
)

resnet_validation_dataset = Subset(
    resnet_full_validation_dataset,
    validation_indices
)

resnet_train_dataloader = DataLoader(
    resnet_train_dataset,
    batch_size=32,
    shuffle=True
)

resnet_validation_dataloader = DataLoader(
    resnet_validation_dataset,
    batch_size=32,
    shuffle=False
)

resnet_test_dataloader = DataLoader(
    resnet_test_dataset,
    batch_size=32,
     shuffle=False
)

print("Classes:", custom_full_train_dataset.classes)
print("Class mapping:", custom_full_train_dataset.class_to_idx)

print("Custom CNN train images:", len(train_dataset))
print("Custom CNN validation images:", len(validation_dataset))
print("Custom CNN test images:", len(test_dataset))

print("ResNet18 train images:", len(resnet_train_dataset))
print("ResNet18 validation images:", len(resnet_validation_dataset))
print("ResNet18 test images:", len(resnet_test_dataset))