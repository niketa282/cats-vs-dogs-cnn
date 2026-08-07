import torch
import torch.nn as nn
from torchvision.models import resnet18
from dataset import resnet_test_dataloader
from tqdm import tqdm

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(f"Using device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


# 1. Build the architecture
model = resnet18(weights=None)

# 2. Make its output layer match the trained model
number_of_features = model.fc.in_features
model.fc = nn.Linear(
    number_of_features,
    2
)

# 3. Load all trained parameters, including pretrained/frozen feature weights
model.load_state_dict(
    torch.load(
        "resnet18_best_model.pth",
        map_location=device
    )
)

model = model.to(device)
# 4. Activate evaluation behaviour
model.eval()


all_labels = []
all_predictions = []

# 5. Perform inference without gradients
with torch.no_grad():
    for images, labels in tqdm(
        resnet_test_dataloader,
        desc="Testing ResNet18"
    ):
        images = images.to(device)
        
        outputs = model(images)
        predicted_classes = outputs.argmax(dim=1)

        all_labels.extend(
            labels.tolist()
        )

        all_predictions.extend(
            predicted_classes.cpu().tolist()
        )


accuracy = accuracy_score(
    all_labels,
    all_predictions
)

confusion = confusion_matrix(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    average="binary",
    zero_division=0
)


print(f"\nResNet18 Test Accuracy: {accuracy * 100:.2f}%")
print(f"ResNet18 Test Precision: {precision:.4f}")
print(f"ResNet18 Test Recall: {recall:.4f}")
print(f"ResNet18 Test F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(confusion)

print("\nClassification Report:")
print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["Cat", "Dog"],
        zero_division=0
    )
)