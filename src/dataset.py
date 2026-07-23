from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from pathlib import Path

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor()
])

train_path = Path("data") / "train"
test_path = Path("data") / "test"

train_dataset = ImageFolder(
    root=train_path,
    transform=train_transform
)

test_dataset = ImageFolder(
    root=test_path,
    transform=test_transform
)

train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False
)

print("Train classes:", train_dataset.classes)
print("Test classes:", test_dataset.classes)

print("Train images:", len(train_dataset))
print("Test images:", len(test_dataset))
print("Total images:", len(train_dataset) + len(test_dataset))

images, labels = next(iter(train_dataloader))
print("Image batch shape:", images.shape) # (number of images, channels, height, width)
print("Label batch shape:", labels.shape)

print(train_dataset.class_to_idx)